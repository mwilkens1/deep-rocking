import torch
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class CNN_trainer():
    """
    Class for loading the images, training a CNN model, plotting scores and 
    testing the model. This is applied to a dataset of electric guitar images 
    from Reverb.com
    """

    def __init__(self):
        self.classes = ['jazzmaster','lespaul', 'mustang', 'prs_se', 'SG',
                        'stratocaster','telecaster']
        self.train_on_gpu = torch.cuda.is_available()
        self.criterion = None
        self.optimizer = None
        self.scores = {}

        if not self.train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')

    def get_loaders(self, batch_size, train_transforms, test_transforms):
        """
        Creates loaders for training, validation and testing data
        
        batch size - integer
        train transforms - transforms object for the training data
        test transforms - transforms object for the validation and testing data
        
        """

        train_data = datasets.ImageFolder('data/train', 
                                            transform=train_transforms)
        validation_data = datasets.ImageFolder('data/validation', 
                                            transform=test_transforms)
        test_data = datasets.ImageFolder('data/test', 
                                            transform=test_transforms)

        self.train_loader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(validation_data, 
                                        batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=batch_size, shuffle=True)

        self.batch_size = batch_size

    def set_model(self, model, path):
        """
        Defines the model
        
        'Model' is the actual CNN
        'path'  is a string for the location (and name) where the best model is
                stored
        
        """
        self.model = model

        if self.train_on_gpu:
            self.model.cuda()

        self.model_path = path

    def train_model(self,max_epochs, stop_at, stopping_criterion):
        """
        Trains the model and keeps score 
        
        max_epochs - int
        stop_at - integer of the number of epochs that the model will continue
                  to run without improvement less than the stopping criterion
        stopping_criterion - float
        
        """

        epoch_count = []
        train_loss_list = []
        valid_loss_list = []
        valid_acc_list = []

        valid_loss_min = np.Inf # track change in validation loss
        stopper = 0 # counter for epochs without improvement in validation data

        for epoch in range(1, max_epochs+1):

            # keep track of training and validation loss
            train_loss = 0.0    
            valid_loss = 0.0
            class_correct = list(0. for i in range(len(self.classes)))
            class_total = list(0. for i in range(len(self.classes)))

            batch = 0

            ###################
            # train the model #
            ###################
            self.model.train()
                
            for data, target in self.train_loader:

                batch+=1

                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
                # Print progress 
                print('\r Epoch: {} \tTraining \t{:.2f}% completed'.format(epoch, batch/len(self.train_loader)*100), end='')
                
            ######################    
            # validate the model #
            ######################

            batch = 0

            self.model.eval()
            for data, target in self.valid_loader:

                batch+=1
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                # convert output probabilities to predicted class
                _, pred = torch.max(output, 1)    
                # compare predictions to true label
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not self.train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
                # calculate test accuracy for each object class
                for i in range(len(target.data)):
                    if len(target.data)>1:
                        label = target.data[i]
                        class_correct[label] += correct[i].item()
                        class_total[label] += 1
                        
                if batch==1:
                    print('\n \t\tValidation \t{:.2f}% completed'.format(batch/len(self.valid_loader)*100), end='')
                else:
                    print('\r \t\tValidation \t{:.2f}% completed'.format(batch/len(self.valid_loader)*100), end='')
            
            # calculate average losses
            train_loss = train_loss/len(self.train_loader.sampler)
            valid_loss = valid_loss/len(self.valid_loader.sampler)
            valid_acc = (100. * np.sum(class_correct) / np.sum(class_total),
                np.sum(class_correct), np.sum(class_total))

            # print training/validation statistics 
            print('\n\t\tTraining Loss: {:.3f} \tValidation Loss: {:.3f}'.format(
                train_loss, valid_loss))
            print('\t\tValidation Accuracy (Overall): %2d%% (%2d/%2d)' % valid_acc)

            epoch_count.append(epoch)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            
            # if loss has decreased
            if  valid_loss < valid_loss_min:
                      
                print('\t\tValidation loss decreased ({:.3f} --> {:.3f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(self.model.state_dict(), self.model_path)
                
                # ... by more than the stopping criterion
                if valid_loss_min - valid_loss > stopping_criterion:
                    stopper = 0 #reset the counter
                
            # If no decrease in loss, or no larger than the stopping criterion
            if valid_loss >= valid_loss_min or valid_loss_min - valid_loss <= stopping_criterion:
                stopper += 1 #counter +1

                print('\t\tDecrease in validation loss less than {} ({})'.format(stopping_criterion, stopper))

                if stopper==stop_at:
                    break
            
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
            
        self.scores = {'epochs': epoch_count,'train loss': train_loss_list, 
                        'valid loss': valid_loss_list, 'valid accuracy': valid_acc_list}

    def plot_scores(self):
        '''Plots the loss and accuracy by epochs ''' 

        accuracy_scores = []
        for x in self.scores['valid accuracy']:
            accuracy_scores.append(x[0])

        plt.figure(figsize=(10, 4))
        
        plt.subplot(1,2,1)
        plt.plot(self.scores['epochs'], self.scores['train loss'], label='Training loss')
        plt.plot(self.scores['epochs'], self.scores['valid loss'], label='Validation loss')
        plt.legend()
        plt.title('Loss in validation set')

        plt.subplot(1,2,2)
        plt.plot(self.scores['epochs'], accuracy_scores)
        plt.title('Accuracy in validation set')

        plt.tight_layout()
        plt.show()

    def test_model(self):
        '''Prints loss and accuracy on testing set'''

        # Load best model
        self.model.load_state_dict(torch.load(self.model_path))

        # track test loss
        test_loss = 0.0
        class_correct = list(0. for i in range(len(self.classes)))
        class_total = list(0. for i in range(len(self.classes)))

        self.model.eval()
        # iterate over test data
        for data, target in self.test_loader:
            
            # move tensors to GPU if CUDA is available
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            # calculate the batch loss
            loss = self.criterion(output, target)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)    
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not self.train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(len(target.data)):
                if len(target.data)>1:
                    label = target.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        # average test loss
        test_loss = test_loss/len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(len(self.classes)):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    self.classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))                