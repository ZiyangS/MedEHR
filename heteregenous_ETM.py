import torch
import torch.nn.functional as F 
import pickle
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class heteregenous_ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, spec_size, t_hidden_size, rho_size, emsize, num_types,
                    theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5,
                    predict_labels=None, multiclass_labels=None, num_labels=None):
        super(heteregenous_ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics # K, 100
        self.vocab_size = vocab_size # V, 3427
        self.spec_size = spec_size # T, 3
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size # L, 300
        self.enc_drop = enc_drop
        self.emsize = emsize# 250
        self.t_drop = nn.Dropout(enc_drop)
        self.num_types = num_types

        self.theta_act = self.get_activation(theta_act)
        self.train_embeddings = train_embeddings
        self.predict_labels = predict_labels
        self.multiclass_labels = multiclass_labels
        self.num_labels = num_labels
        
        ## define the word embedding matrix \rho
        if self.train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
            # self.rho = nn.Parameter(torch.randn(vocab_size, rho_size, num_types)) # V x L X T
            # it will be too much, we do not consider to use rho with types first.
        else:
            self.rho = embeddings.clone().float().to(device)

        ## define the topic embeddings matrix \alpha
        # self.alphas = nn.Linear(rho_size, num_topics, bias=False)#nn.Parameter(torch.randn(rho_size, num_topics))
        ## define the variational parameters for the topic embeddings over time (alpha)
        # alpha is T x K x L
        self.mu_q_alpha = nn.Parameter(torch.randn(num_types, num_topics, rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(num_types, num_topics, rho_size))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size*spec_size, t_hidden_size), # input is V x T for each doc
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)


        if predict_labels:
            self.classifier = nn.Linear(num_topics, num_labels, bias=True)
            self.criterion = nn.BCEWithLogitsLoss(reduction='sum')


    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            act = nn.Tanh()
        return act 


    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu


    def get_beta(self, alpha):
        """
        Returns the topic matrix beta of shape T x K x V
        """
        if self.train_embeddings:
            logit = self.rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)) # logit is K*T x V
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1) # logit is T x K x V now
        beta = F.softmax(logit, dim=-1)  # torch.Size([3, 100, 3427]), T x K x V
        return beta


    def get_theta(self, normalized_bows):
        """
        normalized_bows: batch of normalized bag-of-words, D_minibatch x T x V
        D_minibatch is the document number in a minibatch.
        """
        # reshape normalized_bows to D_minibatch x T*V, we concatenate the different spec's ICD codes
        input = normalized_bows.view(normalized_bows.size(0), normalized_bows.size(1) * normalized_bows.size(2))
        input = input.float()
        q_theta = self.q_theta(input)  # D_minibatch x t_hidden_size
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)  # D_minibatch x K
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta) # D_minibatch x K
        theta = F.softmax(z, dim=-1)  # D_minibatch x K
        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta), we use sum
        # use mean or sum???
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).sum() # scalar value
        # kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return theta, kl_theta


    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """
        Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
            # calculate 1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), it's D X K
            # calculate sum and dim = -1, it's D
        return kl


    def get_alpha(self):  ## mean field
        # alphas = torch.zeros(self.num_types, self.num_topics, self.rho_size).to(device)
        kl_alpha = []
        alphas = self.reparameterize(self.mu_q_alpha, self.logsigma_q_alpha)
        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        for t in range(self.num_types):
            kl_alpha.append(self.get_kl(self.mu_q_alpha[t], self.logsigma_q_alpha[t], p_mu_0, logsigma_p_0))
        # kl_alpha T x K
        # print("kl_alpha")
        # print(kl_alpha[0].size()) # check a kl[t] in type T
        kl_alpha = torch.stack(kl_alpha).sum() # scalar value
        # print(kl_alpha.size())
        return alphas, kl_alpha.sum() # scalar value


    def get_prediction_loss(self, theta, labels):
        if self.multiclass_labels: # multi-class prediction loss as independent Bernoulli
            targets = torch.zeros(theta.size(0), self.num_labels)
            for i in range(theta.size(0)):
                targets[i,labels[i].type('torch.LongTensor').item()] = 1
            labels = targets

        outputs = self.classifier(theta)
        if self.multiclass_labels: # multi-class prediction loss as independent Bernoulli
            pred_loss = (-labels * F.log_softmax(outputs, dim=-1) - (1-labels) * torch.log(1-F.softmax(outputs, dim=-1))).sum()
        else: # single-label prediction
            # pos_cnt = sum(labels)
            # weight = torch.ones(theta.size(0))
            # weight[labels==1] *= theta.size(0) / pos_cnt / 2
            # weight[labels==0] *= theta.size(0) / (theta.size(0)-pos_cnt) / 2
            # pred_loss = F.binary_cross_entropy_with_logits(outputs, labels.type('torch.LongTensor').to(device), pos_weight=self.pos_weight, reduction='sum')
            outputs = outputs.squeeze()
            pred_loss = self.criterion(outputs, labels)
        return pred_loss


    def forward(self, bows, normalized_bows, num_docs_train, labels=None, epoch=0):
        if self.predict_labels:
            bows0, bows1 = bows
            normalized_bows0, normalized_bows1 = normalized_bows
            bsz = normalized_bows0.size(0)
            coeff = num_docs_train / bsz
        else:
            bsz = normalized_bows.size(0)
            coeff = num_docs_train / bsz
        alpha, kl_alpha = self.get_alpha()
        if self.predict_labels:
            theta, kl_theta = self.get_theta(normalized_bows0) # theta is D_minibatch x K
        else:
            theta, kl_theta = self.get_theta(normalized_bows) # theta is D_minibatch x K
        kl_theta = kl_theta * coeff
        beta = self.get_beta(alpha)  # beta is T x K x V

        if self.predict_labels:
            bows0, bows1 = bows
            normalized_bows0, normalized_bows1 = normalized_bows
            theta1, _ = self.get_theta(normalized_bows1)
            pred_loss = self.get_prediction_loss(theta1, labels) * coeff
        else:
            pred_loss = 0

        reshaped_beta = beta.view(beta.size(1), beta.size(0), beta.size(2)) # beta is K x T x V
        reshaped_beta = reshaped_beta.view(reshaped_beta.size(0), reshaped_beta.size(1)*reshaped_beta.size(2)) # beta is K x T*V
        res = torch.mm(theta, reshaped_beta) # D_minibatch x T*V
        res = res.view(res.size(0), beta.size(0), beta.size(2)) # res is D_minibatch x T x V
        preds = torch.log(res + 1e-6)
        if self.predict_labels:
            recon_loss = -(preds * bows0).sum(2).sum(1)  # D, since sum over V and T
        else:
            recon_loss = -(preds * bows).sum(2).sum(1) # D, since sum over V and T
        recon_loss = recon_loss.sum() * coeff # scalar value
        nelbo = recon_loss + kl_alpha + kl_theta + pred_loss
        print(nelbo, recon_loss, kl_alpha, kl_theta)

        return nelbo, recon_loss, kl_alpha, kl_theta, pred_loss

