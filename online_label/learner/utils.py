import torch
from torch import nn, optim
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, n_hidden_layer=0):
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        modules = []
        in_dim = input_dim
        for i in range(n_hidden_layer):
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(nn.Tanh())
            in_dim = hidden_dim
        modules.append(nn.Linear(in_dim, self.output_dim))
        
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        out = self.net(x)
        return out

    def compute_loss(self, logits, labeled_y, reduction='mean'):
        loss = F.cross_entropy(logits, labeled_y, reduction=reduction)
        return loss

    def compute_mixmatch_loss(self, logits_l, prob_l, logits_u, prob_u, u_weight):
        ce_loss = (-F.softmax(logits_l, 1).log() * prob_l).sum(1).mean()
        l2_loss = torch.mean((F.softmax(logits_u, 1) - prob_u)**2)
        return ce_loss + u_weight * l2_loss


class ModelWithTemperature(nn.Module):
    '''
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    '''
    def __init__(self, model, use_cuda=False):
        super(ModelWithTemperature, self).__init__()
        self.use_cuda = use_cuda
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        '''
        Perform temperature scaling on logits
        '''
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    
    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        '''
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        '''
        
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()
        if self.use_cuda:
            nll_criterion = nll_criterion.cuda()
            ece_criterion = ece_criterion.cuda()


        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                if self.use_cuda:
                    input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            if self.use_cuda:
                logits = logits.cuda()
                labels = labels.cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        logger.debug('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        if self.temperature.item() > 100. or self.temperature.item() < -100.:
            logger.debug('Invalid temperature found')
            torch.nn.init.constant_(self.temperature, 1)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        
        logger.debug(f'Optimal temperature: {self.temperature.item()}')
        logger.debug(f'After temperature - NLL: {after_temperature_nll}, ECE:{after_temperature_ece}')
        self.after_nll = after_temperature_nll
        self.before_nll = before_temperature_nll
        self.after_ece = after_temperature_ece
        self.before_ece = before_temperature_ece

        return self
        

class _ECELoss(nn.Module):
    '''
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    'Obtaining Well Calibrated Probabilities Using Bayesian Binning.' AAAI.
    2015.
    '''
    def __init__(self, n_bins=15):
        '''
        n_bins (int): number of confidence interval bins
        '''
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
