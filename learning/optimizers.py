import os
import time
from abc import abstractmethod, ABCMeta
import torch

class Optimizer(metaclass=ABCMeta):
    """Base class for gradient-based optimization algorithms."""

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        Optimizer initializer.
        :param model: Net, the model to be learned.
        :param train_set: DataSet, training set to be used.
        :param evaluator: Evaluator, for computing performance scores during training.
        :param val_set: Datset, validation set to be used, which can be None if not used.
        :param kwargs: dict, extra argument containing training hyperparameters.
                - batch_size: int, batch size for each iteration.
                - num_epochs: int, total number of epochs for training.
                - init_learning_rate: float, initial learning rate.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # Training hyperparameters
        self.batch_size = kwargs.pop('batch_size', 8)
        self.num_epochs = kwargs.pop('num_epochs', 100)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.001)
        self.opt = self._optimize_op()
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self.curr_epoch = 1
        # number of bad epochs, where the model is updated without improvement.
        self.num_bad_epochs = 0
        # initialize best score with the worst one
        self.best_score = self.evaluator.worst_score
        self.curr_learning_rate = self.init_learning_rate

    @abstractmethod
    def _optimize_op(self, **kwargs):
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        pass

    def _step(self):
        """
        Make a single gradient update and return its results.
        This should not be called manually.
        :return loss: float, loss value for the single iteration step.
                y_true: np.ndarray, true label from the training set.
                y_pred: np.ndarray, predicted label from the model.
        """

        # Sample a single batch
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True)
        # Compute the loss and make update
        self.opt.zero_grad()
        X_tensor, y_tensor = torch.tensor(X).cuda(), torch.tensor(y_true).cuda()
        loss = self.model.loss(X_tensor, y_tensor)
        y_pred = self.model.output(X_tensor).cpu().detach().numpy()
        loss.backward()
        self.opt.step()

        return loss, y_true, y_pred

    def train(self, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        Run optimizer to train the model.
        :param sess: tf.Session.
        :param save_dir: str, the directory to save the learned weights of the model.
        :param details: bool, whether to return detailed results.
        :param verbose: bool, whether to print details during training.
        :param kwargs: dict, extra arguments containing training hyperparameters.
                - nms_flag: bool, whether to do non maximum supression(nms) for evaluation.
        :return train_results: dict, containing detailed results of training.
        """
        num_eval = kwargs.pop('num_eval', 128)
        train_results = dict()
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch
        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # Start training loop
        for i in range(num_steps):
            # Perform a gradient update from a single minibatch
            if not self.model.training:
                self.model.train()
            step_loss, step_y_true, step_y_pred = self._step()
            step_losses.append(step_loss)
            # Perform evaluation in the end of each epoch
            if (i+1) % num_eval == 0:
                # Evaluate model with current minibatch, from training set
                step_score = self.evaluator.score(
                    step_y_true, step_y_pred, self.model, **kwargs)
                step_scores.append(step_score)
                # If validation set is initially given, use if for evaluation
                if self.val_set is not None:
                    # Evaluate model with the validation set
                    eval_y_pred = self.model.predict(self.val_set, **kwargs)
                    eval_score = self.evaluator.score(
                        self.val_set.labels, eval_y_pred, self.model, **kwargs)
                    eval_scores.append(eval_score)
                    if verbose:
                        # Print intermediate results
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))

                    curr_score = eval_score
                else:
                    if verbose:
                        # Print intermediate results
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |lr: {:.6f}'
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))

                    curr_score = step_score
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.model.save(os.path.join(save_dir, 'yolov2.pth'))

                # Keep track of the current best model,
                # by comparing current score and the best score
            if (i+1) % num_steps_per_epoch == 0:
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(
                time.time() - start_time))
            print('Best {} score: {}'.format(
                'evaluation' if eval else 'training', self.best_score))

        print('Done.')

        if details:
            # Store training results in a dictionary
            train_results['step_losses'] = step_losses
            train_results['step_scores'] = step_scores
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores

        return train_results

class AdamOptimizer(Optimizer):
    """Gradient descent optimizer, with Momentum algorithm."""

    def _optimize_op(self, **kwargs):
        """
        tf.train.AdamOptimizer.minimize Op for a gradient update.
        :param kwargs: dict, extra arguments for optimizer.
                -momentum: float, the momentum coefficent.
        :return tf.Operation.
        """

        train_op = torch.optim.Adam(self.model.parameters(), lr=self.init_learning_rate)
        return train_op

    def _update_learning_rate(self, **kwargs):
        """
        Update current learning rate, when evaluation score plateaus.
        :param kwargs: dict, extra arguments for learning rate scheduling.
                - learning_rate_patience: int, number of epochs with no improvement after which learning rate will be reduced.
                - learning_rate_decay: float, factor by which the learning rate will be updated.
                -eps: float, if the difference between new and old learning rate is smller than eps, the update is ignored.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # Decay learning rate only when the difference is higher than
            # epsilon.
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
                self._set_learning_rate()
            self.num_bad_epochs = 0

    def _set_learning_rate(self):
        for param_group in self.opt.param_groups:
            param_group['lr'] = self.curr_learning_rate