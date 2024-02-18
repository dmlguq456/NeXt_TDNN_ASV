import os
import re

from tqdm import tqdm
import torch
import glob
import torch.nn.functional as F
import lightning as L
import pandas as pd
import soundfile

from eval_metric import compute_eer, compute_MinDCF
from backend.cosine_similarity_full import cosine_similarity_full
from backend.euclidean_distance_full import euclidean_distance_full



class ENGINE(L.LightningModule):
    """
    contain train, validation, test
        - train: speaker identification
        - validation & test: speaker verification
    Ref1: https://github.com/clovaai/voxceleb_trainer/blob/343af8bc9b325a05bcf28772431b83f5b8817f5a/SpeakerNet.py#L136
    Ref2: https://github.com/zyzisyz/mfa_conformer/blob/master/main.py
    """
    def __init__(self, speaker_net, loss_function, optimizer, scheduler, eval_config, code_save_time, config = None, **kwargs):
        super().__init__()
        self.speaker_net = speaker_net
        self.loss_function = loss_function

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.code_save_time = code_save_time
        self.save_hyperparameters(ignore = ['speaker_net', 'loss_function', 'optimizer', 'scheduler', 'config'])

        # eval_config
        self.eval_config = eval_config
        self.config = config # for score normalization

        # output
        self.validation_step_outputs = []

    # =====================================================
    # ⚡⚡ train
    # =====================================================
    def training_step(self, batch, batch_idx):
        x, y = batch

        # calculate speaker embedding
        speaker_embedding = self.speaker_net(x) # speaker_embedding : (batch, embedding_size)

        # Loss function
        loss, acc = self.loss_function(speaker_embedding, y)

        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True, sync_dist=True)
        self.log("t_Acc", acc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    # ===============================================================
    # ⚡⚡ validation - w/o score normalization
    # ===============================================================
    def validation_step(self, batch, batch_idx):
        """
        During the validation_step, calculate the speaker embedding for all speakers in the test list. 
        Then, during the validation_epoch_end, compute EER & MinDCF using the extracted 'speaker embedding' saved in self.validation_step_outputs.
        """

        x, y = batch # x : (batch_size, audio_length), where batch_size == 1 y: (batch_size, )
               
        assert x.shape[0] == 1, "batch size must be 1"

        # calculate speaker embedding 
        speaker_embedding = self.speaker_net(x) # speaker_embedding_full : (batch, embedding_size)
        speaker_id_to_embedding = {y[0]:speaker_embedding} # y[0]: speaker_id

        self.validation_step_outputs.append(speaker_id_to_embedding) # ['id10309/vobW27_-JyQ/00015.wav': tensor( torch.Size([1, 192]))]

        return speaker_id_to_embedding
    
    def on_validation_epoch_end(self):
        """ 
        Calculate EER & MinDCF using extraceted 'speaker embedding' saved in self.validation_step_outputs.

        Note:
            self.validation_step_outputs,  [speaker_id_to_embedding #1, speaker_id_to_embedding #2, ... , speaker_id_to_embedding #n] : list of dict
                speaker_id_to_embedding (dict)      
                    - key : speaker name
                    - value : speaker embedding, (shape: (batch, embedding_size)), where batch == 1
        """
        # read test list(e.g. veri_test2.txt)        
        test_list = pd.read_csv(self.eval_config['test_list'], sep=' ', header=None, names=['label', 'enroll_filename', 'test_filename'])
                
        # =========================
        # Set variables for eval & compile speaker embeddings into the speaker_embedding_map
        # =========================

        speaker_embedding_map = {} # key: speaker id (e.g., 'id10270/5r0dWxy17C8/00001.wav'), value: speaker embedding (shape: (batch, embedding_size)
        [speaker_embedding_map.update({list(dic.keys())[0] : list(dic.values())[0]}) for dic in self.validation_step_outputs] 

        scores_cosine_list = []
        scores_euclidean_list = []
        labels = []

        # =========================
        # calculate score & EER & MinDCF
        # =========================

        # calculate similarity score 
        for _, row in test_list.iterrows():
            enroll_embedding = speaker_embedding_map[row['enroll_filename']]
            test_embedding = speaker_embedding_map[row['test_filename']]
            labels.append(int(row['label']))

            # compute cosine similarity 
            score_cosine = cosine_similarity_full(enroll_embedding, test_embedding)
            scores_cosine_list.append(score_cosine.detach().cpu().numpy())

            # compute euclidean distance
            score_euclidean = euclidean_distance_full(enroll_embedding, test_embedding)
            scores_euclidean_list.append(score_euclidean.detach().cpu().numpy())

        # calculate EER & MinDCF
        eer_cosine, _ = self.eval_log(scores_cosine_list, labels, 'cosine')
        eer_euclidean, _ = self.eval_log(scores_euclidean_list, labels, 'euclidean')

        print(eer_cosine, eer_euclidean)
        min_eer = min(eer_cosine, eer_euclidean)
        self.log("min_eer", min_eer, on_epoch=True, logger=True, prog_bar=True)

        return
    
    def eval_log(self, scores, labels, log_name=""):
        eer, _ = compute_eer(scores, labels) # calculate EER
        min_dcf, _ = compute_MinDCF(scores, labels, p_target= self.eval_config['p_target'], c_miss = self.eval_config['c_miss'], c_fa = self.eval_config['c_fa']) # calculate MinDCF
        self.log(f"eer_{log_name}", eer, on_epoch=True, logger=True, prog_bar=False)
        self.log(f"mindcf_{log_name}", min_dcf, on_epoch=True, logger=True, prog_bar=False)

        return eer, min_dcf
    

    # ===============================================================
    # ⚡⚡ test - w/ score normalization
    # ===============================================================
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):

        # =========================
        # get cohort
        # =========================
        """
        If extracted cohort is not exist, extract cohort and save it.
        else, load cohort.
        """

        # set cohort save path        
        save_path = self.config.COHORT_SAVE_PATH
        cohort_list_name = os.path.splitext(os.path.basename(self.config.COHORT_LIST_PATH))[0]
        config_name = self.config.TRAINER_CONFIG.get('default_root_dir').split('/')[-1]
        ckpt_epoch = self.find_epoch(self.config.TEST_CHECKPOINT)

        cohort_save_path = f"{save_path}/{config_name}/{cohort_list_name}_{ckpt_epoch}"

        # extract cohort embedding
        if not os.path.exists(cohort_save_path):
            print("⚡" * 10)
            print(f"Extract cohort and save it to {cohort_save_path}")
            print("⚡" * 10)

            # make directory
            os.makedirs(cohort_save_path, exist_ok=True)

            train_path = self.config.TRAIN_DATASET_CONFIG['train_path']

            df = pd.read_csv(self.config.COHORT_LIST_PATH)

            for file_name in tqdm(df['file_name'], desc=f"Extracting imposter features at {cohort_save_path}", total = len(df)):

                # load audio
                file_path = os.path.join(train_path, f"{file_name}.wav")
                audio, sr = soundfile.read(file_path) # audio : (audio_length, )

                assert sr == 16000, f"sample rate must be 16000, but {sr}"

                # extract embedding
                speaker_embedding = self.speaker_net(torch.FloatTensor(audio).unsqueeze(0).to(self.device)) # speaker_embedding : (batch, embedding_size), where batch == 1

                # save embedding
                torch.save(speaker_embedding, f"{cohort_save_path}/{file_name.replace('/', '_').replace('.wav', '')}.pt")

        # load cohort embedding
        cohort_list = []
        imposter_embedding_path = glob.glob(f"{cohort_save_path}/*.pt")

        for path in tqdm(imposter_embedding_path, total = len(imposter_embedding_path), desc=f"Loading imposter features at {cohort_save_path}"):
            cohort_list.append(torch.load(path, map_location=self.device))

        print(f"cohort set are loaded with size: {len(cohort_list)}")

        train_cohort = torch.stack(cohort_list) # train_cohort : (num_impostor, batch, embedding_size)
        train_cohort = train_cohort.squeeze(1) # train_cohort: (num_impostor, embedding_size)
        num_cohort = train_cohort.shape[0]

        # =========================
        # Set variables for eval & compile speaker embeddings into the speaker_embedding_map
        # =========================

        speaker_embedding_map = {} # key: speaker id (e.g., 'id10270/5r0dWxy17C8/00001.wav'), value: speaker embedding (shape: (batch, embedding_size)
        [speaker_embedding_map.update({list(dic.keys())[0] : list(dic.values())[0]}) for dic in self.validation_step_outputs] 

        scores_origin = []
        score_znorm = []
        score_tnorm = []
        score_snorm = []        
        labels = []

        # =========================
        # calculate score
        # =========================

        test_list = pd.read_csv(self.eval_config['test_list'], sep=' ', header=None, names=['label', 'enroll_filename', 'test_filename'])

        for _, row in tqdm(test_list.iterrows(), total=len(test_list), desc=f"Calculate EER with score norm"): # len : 37720

            enroll_embedding = speaker_embedding_map[row['enroll_filename']] # enroll_embedding : (batch, embedding_size), where batch == 1
            test_embedding = speaker_embedding_map[row['test_filename']] # test_embedding : (batch, embedding_size), where batch == 1
            labels.append(int(row['label']))

            # repeat enroll_embedding & calculate similarity score
            enroll_embedding_repeat = enroll_embedding.repeat(num_cohort, 1) # (num_cohort, embedding_size)
            score_enroll_cohort = F.cosine_similarity(enroll_embedding_repeat, train_cohort, dim=1) # (num_cohort, )
            score_enroll_cohort_topk = torch.topk(score_enroll_cohort, k=self.config.TOP_K, dim=0)[0] # (TOP_K, ) 

            # compute mean & std of score_enroll_cohort_topk
            mean_enroll_cohort = torch.mean(score_enroll_cohort_topk, dim=0)
            std_enroll_cohort = torch.std(score_enroll_cohort_topk, dim=0)

            # repeat test_embedding & calculate similarity score
            test_embedding_repeat = test_embedding.repeat(num_cohort, 1)
            score_test_cohort = F.cosine_similarity(test_embedding_repeat, train_cohort, dim=1) 
            score_test_cohort_topk = torch.topk(score_test_cohort, k=self.config.TOP_K, dim=0)[0] 

            # compute mean & std of score_test_cohort_topk
            mean_test_cohort = torch.mean(score_test_cohort_topk, dim=0)
            std_test_cohort = torch.std(score_test_cohort_topk, dim=0)

            # calculate score between enroll_embedding & test_embedding
            score_enroll_test = F.cosine_similarity(enroll_embedding, test_embedding, dim=1) # (num_cohort, )

            # =========================
            # score normalization
            # =========================

            # score z-norm
            score_zn = (score_enroll_test - mean_enroll_cohort) / std_enroll_cohort
            # score t-norm
            score_tn = (score_enroll_test - mean_test_cohort) / std_test_cohort
            # score s-norm
            score_e = (score_enroll_test - mean_enroll_cohort) / std_enroll_cohort
            score_t = (score_enroll_test - mean_test_cohort) / std_test_cohort
            score_sn = 0.5 * (score_e + score_t)

            scores_origin.append(score_enroll_test.detach().cpu().numpy())
            score_znorm.append(score_zn.detach().cpu().numpy())
            score_tnorm.append(score_tn.detach().cpu().numpy())
            score_snorm.append(score_sn.detach().cpu().numpy())

        # =========================
        # calculate EER & MinDCF
        # =========================
        eer_origin, _ = self.eval_log(scores_origin, labels, 'origin')
        eer_znorm, _ = self.eval_log(score_znorm, labels, 'znorm')
        eer_tnorm, _ = self.eval_log(score_tnorm, labels, 'tnorm')
        eer_snorm, _ = self.eval_log(score_snorm, labels, 'snorm')

        min_eer = min(eer_origin, eer_znorm, eer_tnorm, eer_snorm)
        self.log("min_eer", min_eer, on_epoch=True, logger=True, prog_bar=True)
        
        return 
    
    def find_epoch(self, ckpt_path):
        """
        find epoch from ckpt_path
        Args:
            ckpt_path (str): checkpoint path (e.g., 's-epoch=299-loss=4.21-full_minimum=1.72-min_eer_seg=2.61.ckpt'')
        return:
            epoch (str): epoch (e.g., 'epoch=299')
        """        
        p = re.compile('epoch=(\d+)')
        try:
            epoch = p.search(ckpt_path).group()  # 'epoch=299'
        except AttributeError:
            epoch = 'epoch=999' 

        return epoch

    
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "loss",
                "interval": 'epoch',
                "frequency": 1,
                "name": 'lr_log'
            },
        }
    

class ENGINESTEP(ENGINE):
    """
    Only difference is that the learning rate is updated every step.
    """
    def __init__(self, speaker_net, loss_function, optimizer, scheduler, eval_config, code_save_time, config = None, **kwargs):
        super().__init__(speaker_net, loss_function, optimizer, scheduler, eval_config, code_save_time, config, **kwargs)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "loss",
                "interval": 'step',
                "frequency": 1,
                "name": 'lr_log'
            },
        }