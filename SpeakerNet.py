# reference: https://github.com/clovaai/voxceleb_trainer/blob/master/SpeakerNet.py
#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn

class SpeakerNet(nn.Module):
    def __init__(self, feature_extractor, spec_aug, model, aggregation, loss_function, print_model=True):
        super(SpeakerNet, self).__init__()

        # 1. Feature Extraction
        self.feature_extractor = feature_extractor

        # 2. Spec Aug
        self.spec_aug = spec_aug if spec_aug is not None else None
        
        # 3. speaker embedding extractor
        self.model = model

        # 4. aggregate function
        self.aggregate = aggregation

        # 5. loss function
        self.loss_function = loss_function if loss_function is not None else None


        if print_model:
            print("⚡ feature_extractor ⚡")
            print(self.feature_extractor)
            print("⚡ spec_aug ⚡")
            print(self.spec_aug)
            print("⚡ model ⚡")
            print(self.model)
            print("⚡ aggregation ⚡")
            print(self.aggregate)
            print("⚡ loss ⚡")
            print(self.loss_function)

    def forward(self, x):
        """
        Args:
            x: (batch, time)
        Returns:
            x: (batch, embeding_size)
        """
        # 1. Feature Extraction
        x = self.feature_extractor(x)  # (batch, freq, time)

        # 2. Spec Aug
        if self.spec_aug is not None:
            x = self.spec_aug(x) # (batch, freq, time)

        # 3. speaker embedding extractor
        x = self.model(x)  # (batch, channel_size, T)

        # 4. aggregate function
        x = self.aggregate(x) # (batch_size, embeding_size)

        return x # (batch_size, embeding_size)
    
class SpeakerNetMultipleLoss(nn.Module):
    def __init__(self, feature_extractor, spec_aug, model, aggregation, loss_function_metric, loss_function_classification):
        super(SpeakerNetMultipleLoss, self).__init__()

        # 1. Feature Extraction
        self.feature_extractor = feature_extractor
        print("⚡ feature_extractor ⚡")
        print(self.feature_extractor)

        # 2. Spec Aug
        self.spec_aug = spec_aug if spec_aug is not None else None
        print("⚡ spec_aug ⚡")
        print(self.spec_aug)
        
        # 3. speaker embedding extractor
        self.model = model
        print("⚡ model ⚡")
        print(self.model)

        # 4. aggregate function
        self.aggregate = aggregation
        print("⚡ aggregation ⚡")
        print(self.aggregate)

        # 5. loss function: metric learning + classification
        self.loss_function_metric = loss_function_metric if loss_function_metric is not None else None
        print("⚡ loss_function_metric ⚡")
        print(self.loss_function_metric)

        self.loss_function_classification = loss_function_classification if loss_function_classification is not None else None
        print("⚡ loss_function_classification ⚡")
        print(self.loss_function_classification)


    def forward(self, x):
        """
        Args:
            x: (batch, time)
        Returns:
            x: (batch, embeding_size)
        """
        # 1. Feature Extraction
        x = self.feature_extractor(x)  # (batch, freq, time)

        # 2. Spec Aug
        if self.spec_aug is not None:
            x = self.spec_aug(x) # (batch, freq, time)

        # 3. speaker embedding extractor
        x = self.model(x)  # (batch, channel_size, T)

        # 4. aggregate function
        x = self.aggregate(x) # (batch_size, embeding_size)

        return x # (batch_size, embeding_size)



class SpeakerNetWoFeatureEx(nn.Module):
    def __init__(self, model, aggregation, loss_function, spec_aug=None):
        super(SpeakerNetWoFeatureEx, self).__init__()

        # # 1. Feature Extraction
        # self.feature_extractor = feature_extractor
        # print("⚡ feature_extractor ⚡")
        # print(self.feature_extractor)

        # 2. Spec Aug
        self.spec_aug = spec_aug if spec_aug is not None else None
        print("⚡ spec_aug ⚡")
        print(self.spec_aug)
        
        # 3. speaker embedding extractor
        self.model = model
        print("⚡ model ⚡")
        print(self.model)

        # 4. aggregate function
        self.aggregate = aggregation
        print("⚡ aggregation ⚡")
        print(self.aggregate)

        # 5. loss function
        self.loss_function = loss_function if loss_function is not None else None
        print("⚡ loss ⚡")
        print(self.loss_function)


    def forward(self, x):
        """
        Args:
            x: (batch, time)
        Returns:
            x: (batch, embeding_size)
        """
        # 1. Feature Extraction
        # x = self.feature_extractor(x)  # (batch, freq, time)

        # 2. Spec Aug
        if self.spec_aug is not None:
            x = self.spec_aug(x) # (batch, freq, time)

        # 3. speaker embedding extractor
        x = self.model(x)  # (batch, channel_size, T)

        # 4. aggregate function
        x = self.aggregate(x) # (batch_size, embeding_size)

        return x # (batch_size, embeding_size)


# class SpeakerNetMultiLoss(nn.Module):
#     def __init__(self, feature_extractor, spec_aug, model, aggregation, loss_functions:dict, **kwargs):
#         super(SpeakerNet, self).__init__()

#         # 1. Feature Extraction
#         self.feature_extractor = feature_extractor
#         print("⚡ feature_extractor ⚡")
#         print(self.feature_extractor)

#         # 2. Spec Aug
#         self.spec_aug = spec_aug if spec_aug is not None else None
#         print("⚡ spec_aug ⚡")
#         print(self.spec_aug)
        
#         # 3. speaker embedding extractor
#         self.model = model
#         print("⚡ model ⚡")
#         print(self.model)

#         # 4. aggregate function
#         self.aggregate = aggregation
#         print("⚡ aggregation ⚡")
#         print(self.aggregate)

#         # 5. loss function
#         NUM_LOSS = len(loss_functions)
#         self.loss_functions = nn.ModuleList()
#         for i in range(NUM_LOSS):
#             self.loss_functions.append(loss_functions[i])

#         print("⚡ loss ⚡")
#         print(f"number of loss functions: {len(self.loss_functions)}")
#         print(self.loss_functions)


#     def forward(self, x):
#         """
#         Args:
#             x: (batch, time)
#         Returns:
#             x: (batch, embeding_size)
#         """
#         # 1. Feature Extraction
#         x = self.feature_extractor(x)  # (batch, freq, time)

#         # 2. Spec Aug
#         if self.spec_aug is not None:
#             x = self.spec_aug(x) # (batch, freq, time)

#         # 3. speaker embedding extractor
#         x = self.model(x)  # (batch, channel_size, T)

#         # 4. aggregate function
#         x = self.aggregate(x) # (batch_size, embeding_size)

#         return x # (batch_size, embeding_size)
