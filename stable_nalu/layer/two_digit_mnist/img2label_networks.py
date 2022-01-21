import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_nalu.abstract import ExtendedTorchModule


class Img2LabelsSpatialTransformer(ExtendedTorchModule):
    """
    Recreating the MNIST addition network from the Spatial Transformers Paper BUT
    with label clf using concated ST outputs
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformer, self).__init__('stn_clf', **kwargs)
        self.device = device

        self.fc1 = nn.Linear(4*21*21, 200)
        self.fc2 = nn.Linear(200, 128)
        self.fc3 = nn.Linear(128, 20)

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B, 2, 56, 28]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(-1, 20*4*4)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(-1, 2, 3)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 20*4*4)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=28,W=28]
        bsz = x.shape[0]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,56]
        x_st1, x_st2 = self.stn(x)
        x_st = torch.cat([x_st1, x_st2], dim=1)  # [B,4,28,56]

        # Avg pool over concatenated st output
        # [B, 4, 42, 42]
        x_st = F.avg_pool2d(x_st, kernel_size=2)

        # flatten output feature map and pass though FCN to get clf logits [0-19]
        # [B, 4, 21, 21]
        x_st = x_st.view(bsz, 4*21*21)
        x_st = F.relu(self.fc1(x_st))
        x_st = F.relu(self.fc2(x_st))
        x_st = self.fc3(x_st)               # [B,20]

        x_st = x_st.reshape(-1, 2, 10)  # [B,20] -> [B, 2, 10]
        x_st = F.softmax(x_st, dim=-1)  # [B, 2, 10]
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from
        # softargmax
        x_st = x_st @ digits  # [B, 2, 10] [10] = [B,2]

        # return logits [B,20] and spatially transformed channels
        return x_st, x_st1, x_st2

class Img2LabelsSpatialTransformerConcat(ExtendedTorchModule):
    """
    Recreating the MNIST addition network from the Spatial Transformers Paper BUT
    with label clf using concated ST outputs
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformerConcat, self).__init__('stn_clf', **kwargs)
        self.device = device

        self.fc1 = nn.Linear(4*21*21, 200)
        self.fc2 = nn.Linear(200, 128)
        self.fc3 = nn.Linear(128, 20)

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B, 2, 56, 28]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(-1, 20*4*4)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(-1, 2, 3)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 20*4*4)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=28,W=28]
        bsz = x.shape[0]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,56]
        x_st1, x_st2 = self.stn(x)
        x_st = torch.cat([x_st1, x_st2], dim=1)  # [B,4,28,56]

        # Avg pool over concatenated st output
        # [B, 4, 42, 42]
        x_st = F.avg_pool2d(x_st, kernel_size=2)

        # flatten output feature map and pass though FCN to get clf logits [0-19]
        # [B, 4, 21, 21]
        x_st = x_st.view(bsz, 4*21*21)
        x_st = F.relu(self.fc1(x_st))
        x_st = F.relu(self.fc2(x_st))
        x_st = self.fc3(x_st)               # [B,20]

        x_st = x_st.reshape(-1, 2, 10)  # [B,20] -> [B, 2, 10]
        x_st_logits = F.softmax(x_st, dim=-1)  # [B, 2, 10]
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from
        # softargmax
        x_st_pred = x_st_logits @ digits  # [B, 2, 10] [10] = [B,2]

        # return logits [B,20] and spatially transformed channels
        return x_st_pred, x_st1, x_st2

class Img2LabelsSpatialTransformerConvConcat(ExtendedTorchModule):
    """
    2 Spatial Transformer outputs concatenated (4 channel) and put through a convolutional classifier layer.
    Output will be the guess f.e. digit label.
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformerConvConcat, self).__init__('stn_clf', **kwargs)
        self.device = device

        # classifier once ST output is given
        self.st_label_clf = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(4096, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 20),
        )

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B, 2, 56, 28]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(-1, 20*4*4)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(-1, 2, 3)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 20*4*4)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=28,W=28]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,42,42]
        x_st1, x_st2 = self.stn(x)
        x_st = torch.cat([x_st1, x_st2], dim=1)  # [B,4,42,42]

        # Avg pool over concatenated st output
        # [B, 4, 42, 42] -> [B,4,21,21]
        x_st = F.avg_pool2d(x_st, kernel_size=2)
        x_st = self.st_label_clf(x_st)        # [B,20]
        x_st = x_st.reshape(-1, 2, 10)  # [B,20] -> [B, 2, 10]
        x_st_logits = F.softmax(x_st, dim=-1)  # [B, 2, 10]
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from
        # softargmax
        x_st_pred = x_st_logits @ digits  # [B, 2, 10] [10] = [B,2]

        # return logits [B,2] and spatially transformed channels
        return x_st_pred, x_st1, x_st2


class Img2LabelsSpatialTransformerLinearNoConcat(ExtendedTorchModule):
    """
    Recreating the MNIST addition network from the Spatial Transformers Paper BUT
    with a label clf after the STNs (using a non-concat method)
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformerLinearNoConcat, self).__init__('stn_clf', **kwargs)
        self.device = device

        self.fc1 = nn.Linear(2*21*21, 200)
        self.fc2 = nn.Linear(200, 128)
        # self.fc3 = nn.Linear(128, 10)
        # 1 f.e. ST output to clf 1 digit
        self.fc3a = nn.Linear(128, 10)
        self.fc3b = nn.Linear(128, 10)

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B, 2, 56, 28]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(-1, 20*4*4)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(-1, 2, 3)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 20*4*4)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=28,W=28]
        bsz = x.shape[0]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,28]
        x_st1, x_st2 = self.stn(x)
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from

        # predict digit 1
        z_st1 = F.avg_pool2d(x_st1, kernel_size=2)
        z_st1 = z_st1.view(bsz, 2*21*21)
        z_st1 = F.relu(self.fc1(z_st1))
        z_st1 = F.relu(self.fc2(z_st1))
        z_st1 = self.fc3a(z_st1)               # [B,10]
        z_st1 = F.softmax(z_st1, dim=-1)
        z_st1 = z_st1 @ digits  # [B, 2, 10] [10] = [B,1]

        # predict digit 2
        z_st2 = F.avg_pool2d(x_st2, kernel_size=2)
        z_st2 = z_st2.view(bsz, 2*21*21)
        z_st2 = F.relu(self.fc1(z_st2))
        z_st2 = F.relu(self.fc2(z_st2))
        z_st2 = self.fc3b(z_st2)               # [B,10]
        z_st2 = F.softmax(z_st2, dim=-1)
        z_st2 = z_st2 @ digits  # [B, 2, 10] [10] = [B,1]

        # [B,2]
        output = torch.stack((z_st1, z_st2), dim=1)
        return output, x_st1, x_st2

class Img2LabelsSpatialTransformerConvNoConcat(ExtendedTorchModule):
    """
    Recreating the MNIST addition network from the Spatial Transformers Paper BUT
    with a label clf after the STNs (using a non-concat method)
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformerConvNoConcat, self).__init__('stn_clf', **kwargs)
        self.device = device

        # classifier once ST output is given
        self.st_fcn = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(4096, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B, 2, 42, 42] -> xs1 = [B,20, 4,4]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(-1, 20*4*4)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(-1, 2, 3)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 20*4*4)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=28,W=28]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,28]
        x_st1, x_st2 = self.stn(x)
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from

        # predict digit 1
        z_st1 = F.avg_pool2d(x_st1, kernel_size=2)
        z_st1 = self.st_fcn(z_st1)
        z_st1 = F.softmax(z_st1, dim=-1)
        z_st1 = z_st1 @ digits  # [B, 2, 10] [10] = [B,1]

        # predict digit 2
        z_st2 = F.avg_pool2d(x_st2, kernel_size=2)
        z_st2 = self.st_fcn(z_st2)
        z_st2 = F.softmax(z_st2, dim=-1)
        z_st2 = z_st2 @ digits  # [B, 2, 10] [10] = [B,1]

        # [B,2]
        output = torch.stack((z_st1, z_st2), dim=1)
        return output, x_st1, x_st2

class Img2LabelsSpatialTransformerIndepConvNoConcat(ExtendedTorchModule):
    """
    Recreating the MNIST addition network from the Spatial Transformers Paper BUT
    with a conv label clf after the STNs (using a non-concat method). Each ST output has it's own label clf (indep).
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsSpatialTransformerIndepConvNoConcat, self).__init__('stn_clf', **kwargs)
        self.device = device

        # classifier once ST1 output is given
        self.st_fcn1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(4096, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # classifier once ST2 output is given
        self.st_fcn2 = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(4096, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(2, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(320, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        # x = [B, 2, 56, 28]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(-1, 20*4*4)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(-1, 2, 3)
        # [B,H,W,2] = affine_grid([B,2,3], [B, 2, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=2, H=42, W=42] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(-1, 20*4*4)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(-1, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=28,W=28]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,28,28]
        x_st1, x_st2 = self.stn(x)
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from

        # predict digit 1
        z_st1 = F.avg_pool2d(x_st1, kernel_size=2)
        z_st1 = self.st_fcn1(z_st1)
        z_st1 = F.softmax(z_st1, dim=-1)
        z_st1 = z_st1 @ digits  # [B, 2, 10] [10] = [B,1]

        # predict digit 2
        z_st2 = F.avg_pool2d(x_st2, kernel_size=2)
        z_st2 = self.st_fcn2(z_st2)
        z_st2 = F.softmax(z_st2, dim=-1)
        z_st2 = z_st2 @ digits  # [B, 2, 10] [10] = [B,1]

        # [B,2]
        output = torch.stack((z_st1, z_st2), dim=1)
        return output, x_st1, x_st2


class Img2LabelsWidthSpatialTransformerConvNoConcat(ExtendedTorchModule):
    """
    Apply Spatial Transformers Paper with a conv label clf after the STNs (using a non-concat method)
    The 2 MNIST images  will be concatenated on the width channel not the colour channel
    """
    def __init__(self, device, **kwargs):
        super(Img2LabelsWidthSpatialTransformerConvNoConcat, self).__init__('stn_clf', **kwargs)
        self.device = device

        # classifier once ST output is given
        self.st_fcn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.25),
            nn.Flatten(1),
            nn.Linear(9728, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # Spatial transformer localization-network
        self.loc1 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc1 = nn.Sequential(
            nn.Linear(20*4*15, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Spatial transformer localization-network
        self.loc2 = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 20, kernel_size=5),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc2 = nn.Sequential(
            nn.Linear(20*4*15, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc1[2].weight.data.zero_()
        self.fc_loc1[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.fc_loc2[2].weight.data.zero_()
        self.fc_loc2[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        bsz = x.shape[0]
        # x = [B, 1, 42, 84]
        xs1 = self.loc1(x)
        # [B, 320]
        xs1 = xs1.view(bsz, 20*4*15)
        # [B, 6]
        theta1 = self.fc_loc1(xs1)
        # [B, 2, 3]
        theta1 = theta1.view(bsz, 2, 3)
        # [B,H,W,1] = affine_grid([B,2,3], [B, 1, H, W])
        grid1 = F.affine_grid(theta1, x.size(), align_corners=False)
        # [B, C=1, H=42, W=84] = grid_sample([B, 2, H, W], [B,H,W,2])
        x1 = F.grid_sample(x, grid1, align_corners=False)

        xs2 = self.loc2(x)
        xs2 = xs2.view(bsz, 20*4*15)
        theta2 = self.fc_loc2(xs2)
        theta2 = theta2.view(bsz, 2, 3)
        grid = F.affine_grid(theta2, x.size(), align_corners=False)
        x2 = F.grid_sample(x, grid, align_corners=False)

        return x1, x2

    def forward(self, x):
        # x -> [B,2,H=42,W=42]
        # transform the input where each return tensor has shape [B,C,H,W] = [B,2,42,42]
        x_st1, x_st2 = self.stn(x)
        digits = torch.arange(0, 10, 1).type(torch.FloatTensor).to(self.device)  # [10] numbers to index from

        # predict digit 1
        z_st1 = F.avg_pool2d(x_st1, kernel_size=2)
        z_st1 = self.st_fcn(z_st1)
        z_st1 = F.softmax(z_st1, dim=-1)
        z_st1 = z_st1 @ digits  # [B, 2, 10] [10] = [B,1]

        # predict digit 2
        z_st2 = F.avg_pool2d(x_st2, kernel_size=2)
        z_st2 = self.st_fcn(z_st2)
        z_st2 = F.softmax(z_st2, dim=-1)
        z_st2 = z_st2 @ digits  # [B, 2, 10] [10] = [B,1]

        # [B,2]
        output = torch.stack((z_st1, z_st2), dim=1)
        return output, x_st1, x_st2
