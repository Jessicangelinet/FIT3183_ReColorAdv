""" Holds the various attacks we can do """
from __future__ import print_function
from six import string_types
import torch
from torch.autograd import Variable, Function
from torch import optim


from .utils import pytorch_utils as utils
from .utils import image_utils as img_utils
import random
import sys
from . import loss_functions as lf
from . import spatial_transformers as st
import torch.nn as nn
from . import adversarial_perturbations as ap
MAXFLOAT = 1e20


###############################################################################
#                                                                             #
#                      PARENT CLASS FOR ADVERSARIAL ATTACKS                   #
#                                                                             #
###############################################################################

class AdversarialAttack(object):
    """ Wrapper for adversarial attacks. Is helpful for when subsidiary methods
        are needed.
    """

    def __init__(self, classifier_net, normalizer, threat_model,
                 manual_gpu=None):
        """ Initializes things to hold to perform a single batch of
            adversarial attacks
        ARGS:
            classifier_net : nn.Module subclass - neural net that is the
                             classifier we're attacking
            normalizer : DifferentiableNormalize object - object to convert
                         input data to mean-zero, unit-var examples
            threat_model : ThreatModel object - object that allows us to create
                           per-minibatch adversarial examples
            manual_gpu : None or boolean - if not None, we override the
                         environment variable 'MISTER_ED_GPU' for how we use
                         the GPU in this object

        """
        self.classifier_net = classifier_net
        self.normalizer = normalizer or utils.IdentityNormalize()
        if manual_gpu is not None:
            self.use_gpu = manual_gpu
        else:
            self.use_gpu = utils.use_gpu()
        self.validator = lambda *args: None
        self.threat_model = threat_model

    @property
    def _dtype(self):
        return torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

    def setup(self):
        self.classifier_net.eval()
        self.normalizer.differentiable_call()


    def eval(self, ground_examples, adversarials, labels, topk=1):
        """ Evaluates how good the adversarial examples are
        ARGS:
            ground_truths: Variable (NxCxHxW) - examples before we did
                           adversarial perturbation. Vals in [0, 1] range
            adversarials: Variable (NxCxHxW) - examples after we did
                           adversarial perturbation. Should be same shape and
                           in same order as ground_truth
            labels: Variable (longTensor N) - correct labels of classification
                    output
        RETURNS:
            tuple of (% of correctly classified original examples,
                      % of correctly classified adversarial examples)
        """
        normed_ground = self.normalizer.forward(ground_examples)
        ground_output = self.classifier_net.forward(normed_ground)

        normed_advs = self.normalizer.forward(adversarials)
        adv_output = self.classifier_net.forward(normed_advs)

        start_prec = utils.accuracy(ground_output.data, labels.data,
                                    topk=(topk,))
        adv_prec = utils.accuracy(adv_output.data, labels.data,
                                  topk=(topk,))

        return float(start_prec[0]), float(adv_prec[0])


    def eval_attack_only(self, adversarials, labels, topk=1):
        """ Outputs the accuracy of the adv_inputs only
        ARGS:
            adv_inputs: Variable NxCxHxW - examples after we did adversarial
                                           perturbation
            labels: Variable (longtensor N) - correct labels of classification
                                              output
            topk: int - criterion for 'correct' classification
        RETURNS:
            (int) number of correctly classified examples
        """

        normed_advs = self.normalizer.forward(adversarials)

        adv_output = self.classifier_net.forward(normed_advs)
        return utils.accuracy_int(adv_output, labels, topk=topk)



    def print_eval_str(self, ground_examples, adversarials, labels, topk=1):
        """ Prints how good this adversarial attack is
            (explicitly prints out %CorrectlyClassified(ground_examples)
            vs %CorrectlyClassified(adversarials)

        ARGS:
            ground_truths: Variable (NxCxHxW) - examples before we did
                           adversarial perturbation. Vals in [0, 1] range
            adversarials: Variable (NxCxHxW) - examples after we did
                           adversarial perturbation. Should be same shape and
                           in same order as ground_truth
            labels: Variable (longTensor N) - correct labels of classification
                    output
        RETURNS:
            None, prints some stuff though
        """

        og, adv = self.eval(ground_examples, adversarials, labels, topk=topk)
        print("Went from %s correct to %s correct" % (og, adv))



    def validation_loop(self, examples, labels, iter_no=None):
        """ Prints out validation values interim for use in iterative techniques
        ARGS:
            new_examples: Variable (NxCxHxW) - [0.0, 1.0] images to be
                          classified and compared against labels
            labels: Variable (longTensor
            N) - correct labels for indices of
                             examples
            iter_no: String - an extra thing for prettier prints
        RETURNS:
            None
        """
        normed_input = self.normalizer.forward(examples)
        new_output = self.classifier_net.forward(normed_input)
        new_prec = utils.accuracy(new_output.data, labels.data, topk=(1,))
        print_str = ""
        if isinstance(iter_no, int):
            print_str += "(iteration %02d): " % iter_no
        elif isinstance(iter_no, string_types):
            print_str += "(%s): " % iter_no
        else:
            pass

        print_str += " %s correct" % float(new_prec[0])

        print(print_str)



##############################################################################
#                                                                            #
#                            CARLINI WAGNER                                  #
#                                                                            #
##############################################################################
"""
General class of CW attacks: these aim to solve optim problems of the form

Adv(x) = argmin_{x'} D(x, x')
    s.t. f(x) != f(x')
         x' is a valid attack (e.g., meets LP bounds)

Which is typically relaxed to solving
Adv(x) = argmin_{x'} D(x, x') + lambda * L_adv(x')
where L_adv(x') is only nonpositive when f(x) != f(x').

Binary search is performed on a per-example basis to find the appropriate
lambda.

The distance function is backpropagated through in each bin search step, so it
needs to be differentiable. It does not need to be a true distance metric tho
"""


class CarliniWagner(AdversarialAttack):

    def __init__(self, classifier_net, normalizer, threat_model,
                 distance_fxn, carlini_loss, manual_gpu=None):
        """ This is a different init than the other style attacks because the
            loss function is separated into two arguments here
        ARGS:
            classifier_net: standard attack arg
            normalizer: standard attack arg
            threat_model: standard attack arg
            distance_fxn: lf.ReferenceRegularizer subclass (CLASS NOT INSTANCE)
                         - is a loss function
                          that stores originals so it can be used to create a
                          RegularizedLoss object with the carlini loss object
            carlini_loss: lf.PartialLoss subclass (CLASS NOT INSTANCE) - is the
                          loss term that is
                          a function on images and labels that only
                          returns zero when the images are adversarial
        """
        super(CarliniWagner, self).__init__(classifier_net, normalizer,
                                            threat_model, manual_gpu=manual_gpu)


        assert issubclass(distance_fxn, lf.ReferenceRegularizer)
        assert issubclass(carlini_loss, lf.CWLossF6)

        self.loss_classes = {'distance_fxn': distance_fxn,
                             'carlini_loss': carlini_loss}

        self.iteration = 0 #🌸 momentum


    def _construct_loss_fxn(self, initial_lambda, confidence):
        """ Uses the distance_fxn and carlini_loss to create a loss function to
            be optimized
        ARGS:
            initial_lambda : float - which lambda to use initially
                             in the regularization of the carlini loss
            confidence : float - how great the difference in the logits must be
                                 for the carlini_loss to be zero. Overwrites the
                                 self.carlini_loss.kappa value
        RETURNS:
            RegularizedLoss OBJECT to be used as the loss for this optimization
        """
        losses = {'distance_fxn': self.loss_classes['distance_fxn'](None,
                                                       use_gpu=self.use_gpu ),
                  'carlini_loss': self.loss_classes['carlini_loss'](
                                                    self.classifier_net,
                                                    self.normalizer,
                                                    kappa=confidence)}
        scalars = {'distance_fxn': 1.0,
                   'carlini_loss': initial_lambda}
        return lf.RegularizedLoss(losses, scalars)


    def _optimize_step(self, optimizer, perturbation, var_examples,
                       var_targets, var_scale, loss_fxn, targeted=False,
                       momentum=0.9, grad_accum_steps=5):
        """ Does one step of optimization """
        assert not targeted
        optimizer.zero_grad()

        loss = loss_fxn.forward(perturbation(var_examples), var_targets)
        if torch.numel(loss) > 1:
            loss = loss.sum()
        loss.backward()

        # 🌸FIT3183 CHANGE
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(perturbation.parameters(), max_norm=1.0)

        # Gradient accumulation
        if hasattr(self, 'grad_accum'):
            self.grad_accum = momentum * self.grad_accum + perturbation.grad
        else:
            self.grad_accum = perturbation.grad

        if self.iteration % grad_accum_steps == 0:
            for param in perturbation.parameters():
                param.grad = self.grad_accum
            optimizer.step()
            self.grad_accum = None

        self.iteration += 1

        
        # return a loss 'average' to determine if we need to stop early
        return loss.item()


    def _batch_compare(self, example_logits, targets, confidence=0.0,
                       targeted=False):
        """ Returns a list of indices of valid adversarial examples
        ARGS:
            example_logits: Variable/Tensor (Nx#Classes) - output logits for a
                            batch of images
            targets: Variable/Tensor (N) - each element is a class index for the
                     target class for the i^th example.
            confidence: float - how much the logits must differ by for an
                                attack to be considered valid
            targeted: bool - if True, the 'targets' arg should be the targets
                             we want to hit. If False, 'targets' arg should be
                             the targets we do NOT want to hit
        RETURNS:
            Variable ByteTensor of length (N) on the same device as
            example_logits/targets  with 1's for successful adversaral examples,
            0's for unsuccessful
        """
        # check if the max val is the targets
        target_vals = example_logits.gather(1, targets.view(-1, 1))
        max_vals, max_idxs = torch.max(example_logits, 1)
        max_eq_targets = torch.eq(targets, max_idxs)

        # check margins between max and target_vals
        if targeted:
            max_2_vals, _ = example_logits.kthvalue(2, dim=1)
            good_confidence = torch.gt(max_vals - confidence, max_2_vals)
            one_hot_indices = max_eq_targets * good_confidence
        else:
            good_confidence = torch.gt(max_vals.view(-1, 1),
                                       target_vals + confidence)
            # 🌸 FIT3183 changes
            one_hot_indices = (torch.logical_not(max_eq_targets).view(-1, 1) * good_confidence)
            # one_hot_indices = ((1 - max_eq_targets).view(-1, 1) * good_confidence)

        return one_hot_indices.squeeze()
        # return [idx for idx, el in enumerate(one_hot_indices) if el[0] == 1]

    @classmethod
    def tweak_lambdas(cls, var_scale_lo, var_scale_hi, var_scale,
                      successful_mask):
        """ Modifies the constant scaling that we keep to weight f_adv vs D(.)
            in our loss function.

                IF the attack was successful
                THEN hi -> lambda
                     lambda -> (lambda + lo) /2
                ELSE
                     lo -> lambda
                     lambda -> (lambda + hi) / 2


        ARGS:
            var_scale_lo : Variable (N) - variable that holds the running lower
                           bounds in our binary search
            var_scale_hi: Variable (N) - variable that holds the running upper
                          bounds in our binary search
            var_scale : Variable (N) - variable that holds the lambdas we
                        actually use
            successful_mask : Variable (ByteTensor N) - mask that holds the
                              indices of the successful attacks
        RETURNS:
            (var_scale_lo, var_scale_hi, var_scale) but modified according to
            the rule describe in the spec of this method
        """
        prev_his = var_scale_hi.data
        downweights = (var_scale_lo.data + var_scale.data) / 2.0
        upweights = (var_scale_hi.data + var_scale.data) / 2.0

        scale_hi = utils.fold_mask(var_scale.data, var_scale_hi.data,
                                   successful_mask.data)
        scale_lo = utils.fold_mask(var_scale_lo.data, var_scale.data,
                                   successful_mask.data)
        scale = utils.fold_mask(downweights, upweights,
                                successful_mask.data)

        # Logging for debugging
        print(f"tweak_lambdas: scale_lo={scale_lo.mean().item()}, scale_hi={scale_hi.mean().item()}, scale={scale.mean().item()}")
        print(f"tweak_lambdas: successful_mask={successful_mask}")

        
        return (Variable(scale_lo), Variable(scale_hi), Variable(scale))


    def attack(self, examples, labels, targets=None, initial_lambda=1.0,
               num_bin_search_steps=10, num_optim_steps=1000,
               confidence=0.0, verbose=True,
               momentum=0.07, grad_accum_steps=1):
        """ Performs Carlini Wagner attack on provided examples to make them
            not get classified as the labels.
        ARGS:
            examples : Tensor (NxCxHxW) - input images to be made adversarial
            labels : Tensor (N) - correct labels of the examples
            initial_lambda : float - which lambda to use initially
                             in the regularization of the carlini loss
            num_bin_search_steps : int - how many binary search steps we perform
                                   to optimize the lambda
            num_optim_steps : int - how many optimizer steps we perform during
                                    each binary search step (we may stop early)
            confidence : float - how great the difference in the logits must be
                                 for the carlini_loss to be zero. Overwrites the
                                 self.carlini_loss.kappa value
        RETURNS:
            AdversarialPerturbation object with correct parameters.
            Calling perturbation() gets Variable of output and
            calling perturbation().data gets tensor of output
            calling perturbation(distances=True) returns a dict like
                {}
        """

        ######################################################################
        #   First perform some setups                                        #
        ######################################################################

        if targets is not None:
            raise NotImplementedError("Targeted attacks aren't built yet")

        if self.use_gpu:
            examples = examples.cuda()
            labels = labels.cuda()

        self.classifier_net.eval() # ALWAYS EVAL FOR BUILDING ADV EXAMPLES

        var_examples = Variable(examples, requires_grad=False)
        var_labels = Variable(labels, requires_grad=False)


        loss_fxn = self._construct_loss_fxn(initial_lambda, confidence)
        loss_fxn.setup_attack_batch(var_examples)
        distance_fxn = loss_fxn.losses['distance_fxn']

        num_examples = examples.shape[0]

        best_results = {'best_dist': torch.ones(num_examples)\
                                                 .type(examples.type()) \
                                                   * MAXFLOAT,
                        'best_perturbation': self.threat_model(examples)}



        ######################################################################
        #   Now start the binary search                                      #
        ######################################################################
        var_scale_lo = Variable(torch.zeros(num_examples)\
                                .type(self._dtype).squeeze())


        var_scale = Variable(torch.ones(num_examples, 1).type(self._dtype) *
                             initial_lambda).squeeze()
        var_scale_hi = Variable(torch.ones(num_examples).type(self._dtype)
                                * 128).squeeze() # HARDCODED UPPER LIMIT


        for bin_search_step in range(num_bin_search_steps):
            perturbation = self.threat_model(examples)
            ##################################################################
            #   Optimize with a given scale constant                         #
            ##################################################################
            if verbose:
                print("Starting binary_search_step %02d..." % bin_search_step)

            prev_loss = MAXFLOAT
            optimizer = optim.Adam(perturbation.parameters(), lr=0.001)

            for optim_step in range(num_optim_steps):

                if verbose and optim_step > 0 and optim_step % 25 == 0:
                    print("Optim search: %s, Loss: %s" %
                          (optim_step, prev_loss))

                loss_sum = self._optimize_step(optimizer, perturbation,
                                               var_examples, var_labels,
                                               var_scale, loss_fxn,
                                            #    🌸 FIT3183 CHANGES
                                               momentum=momentum, 
                                               grad_accum_steps=grad_accum_steps)

                if loss_sum + 1e-10 > prev_loss * 0.99999 and optim_step >= 100:
                    if verbose:
                        print(("...stopping early on binary_search_step %02d "
                               " after %03d iterations" ) % (bin_search_step,
                                                             optim_step))
                    break
                prev_loss = loss_sum
            # End inner optimize loop

            ################################################################
            #   Update with results from optimization                      #
            ################################################################

            # We only keep this round's perturbations if two things occur:
            # 1) the perturbation fools the classifier
            # 2) the perturbation is closer to original than the best-so-far


            bin_search_perts = perturbation(var_examples)
            bin_search_out = self.classifier_net.forward(bin_search_perts)
            successful_attack_idxs = self._batch_compare(bin_search_out,
                                                         var_labels)


            batch_dists = distance_fxn.forward(bin_search_perts).data

            successful_dist_idxs = batch_dists < best_results['best_dist']
            successful_dist_idxs = successful_dist_idxs


            successful_mask = successful_attack_idxs * successful_dist_idxs


            
            # Logging for debugging
            print(f"successful_attack_idxs: {successful_attack_idxs}")
            print(f"batch_dists: {batch_dists}")
            print(f"successful_dist_idxs: {successful_dist_idxs}")
            print(f"successful_mask: {successful_mask}")


            # And then generate a new 'best distance' and 'best perturbation'

            best_results['best_dist'] = utils.fold_mask(batch_dists,
                                                      best_results['best_dist'],
                                                      successful_mask)

            best_results['best_perturbation'] =\
                 perturbation.merge_perturbation(
                                              best_results['best_perturbation'],
                                              successful_mask)

            # And then adjust the scale variables (lambdas)
            new_scales = self.tweak_lambdas(var_scale_lo, var_scale_hi,
                                            var_scale,
                                            Variable(successful_mask))

            var_scale_lo, var_scale_hi, var_scale = new_scales
            
            # 🌸 FIT3183 changes DEBUG
            if verbose:
                print(f"Updated lambdas: lo={var_scale_lo.mean().item()}, hi={var_scale_hi.mean().item()}, scale={var_scale.mean().item()}")


        # End binary search loop
        if verbose:
            num_successful = len([_ for _ in best_results['best_dist']
                                  if _ < MAXFLOAT])
            print("\n Ending attack")
            print("Successful attacks for %03d/%03d examples in CONTINUOUS" %\
                  (num_successful, num_examples))

        loss_fxn.cleanup_attack_batch()
        perturbation.attach_originals(examples)
        perturbation.attach_attr('distances', best_results['best_dist'])

        return perturbation


    def evaluate(self, examples, labels):
        '''
        🌸 FIT3183 CHANGES
        '''
        perturbation = self.attack(examples, labels)
        perturbation_modified = self.attack(examples, labels, momentum=0.09, grad_accum_steps=2)

        l2_norm = torch.norm(perturbation(examples) - examples)
        l_inf_norm = torch.max(torch.abs(perturbation(examples) - examples))

        l2_norm_modified = torch.norm(perturbation_modified(examples) - examples)
        l_inf_norm_modified = torch.max(torch.abs(perturbation_modified(examples) - examples))

        print(f"Original Carlini-Wagner: L2 Norm = {l2_norm}, L∞ Norm = {l_inf_norm}")
        print(f"Revised Carlini-Wagner: L2 Norm = {l2_norm_modified}, L∞ Norm = {l_inf_norm_modified}")
        print(f"Absolute Differences: L2 Norm = {abs(l2_norm - l2_norm_modified)}, L∞ Norm = {abs(l_inf_norm - l_inf_norm_modified)}")

        # Use the parent/wrapper class's print_eval_str method to check accuracy        
        print("Original Carlini-Wagner Evaluation:")
        self.print_eval_str(examples, perturbation(examples), labels)
        # Evaluate the performance
        original_accuracy, adversarial_accuracy = self.eval(examples, perturbation(examples), labels)
        print(f"Original Accuracy: {original_accuracy}%")
        print(f"Adversarial Accuracy: {adversarial_accuracy}%")
        # Collect and display successful adversarial examples
        successful_advs, successful_origs = perturbation.collect_successful(self.classifier_net, self.normalizer)
        successful_diffs = ((successful_advs - successful_origs) * 3 + 0.5).clamp(0, 1)
        img_utils.show_images([successful_origs, successful_advs, successful_diffs])
        
        print("\nRevised Carlini-Wagner Evaluation:")
        self.print_eval_str(examples, perturbation_modified(examples), labels)
        # Evaluate the performance
        original_accuracy, adversarial_accuracy = self.eval(examples, perturbation_modified(examples), labels)
        print(f"Original Accuracy: {original_accuracy}%")
        print(f"Adversarial Accuracy: {adversarial_accuracy}%")
        # Collect and display successful adversarial examples
        successful_advs, successful_origs = perturbation_modified.collect_successful(self.classifier_net, self.normalizer)
        successful_diffs = ((successful_advs - successful_origs) * 3 + 0.5).clamp(0, 1)
        img_utils.show_images([successful_origs, successful_advs, successful_diffs])