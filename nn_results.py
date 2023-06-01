from tensorflow.keras.models import Model
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import os

from saliency_maps.integrated_gradients import integrated_gradients


class GraphPlotter(object):

    def __init__(self, save_dir, model_id):
        self.save_dir = save_dir
        self.model_id = model_id

    def save_plot(self, filename, directory=None, kwargs={}):
        if directory:
            os.makedirs(os.path.join(self.save_dir, directory), exist_ok=True)
            filename = '{}/{}'.format(directory, filename)
        plt.savefig(os.path.join(self.save_dir, filename + '_{}.png'.format(self.model_id)), **kwargs)
        plt.close()

    def plot_training_graphs(self, history):
        self.plot_mse_loss_history(history, mode='loss', label='Loss')
        self.plot_da_loss_history(history, mode='loss', label='Loss')

    def plot_vae_training_graphs(self, history):
        self.plot_vae_loss_history(history, mode='loss', label='Loss')

    def plot_flow_training_graphs(self, history):
        self.plot_flow_loss_history(history, mode='loss', label='Loss')

    def plot_evaluation_results(self, y_true, y_pred, prior, y_pred_distr=None, mdn=True):
        self.plot_prediction_vs_true(y_true[:128], y_pred[:128])
        self.plot_with_percentiles(y_true, y_pred)

        if mdn:
            y_pred = y_pred_distr.mean().numpy().reshape(-1)
            y_pred_std = y_pred_distr.stddev().numpy().reshape(-1)
            self.plot_prediction_vs_true_with_error_bars(y_true[:128], y_pred[:128], y_pred_std[:128])
            self.plot_prediction_vs_true_with_error_bars_smooth(y_true[:128], y_pred[:128], y_pred_std[:128])
            self.plot_with_median(y_true[:128], y_pred[:128])

            y_pred_prior_mean, y_pred_mode = self.calculate_corrected_y_pred(y_true, y_pred_distr, prior)
            self.plot_with_percentiles(y_true, y_pred_prior_mean, corrected=1)
            self.plot_with_percentiles(y_true, y_pred_mode, corrected=2)
            self.plot_prediction_vs_true_with_error_bars(y_true[:128], y_pred_prior_mean[:128], y_pred_std[:128],
                                                         correction=True)

    def plot_mse_loss_history(self, history, mode='loss', label='Loss'):
        plt.figure()

        epochs = len(history.history[mode][3:])
        plt.plot(range(epochs), history.history[mode][3:], label=label)
        val_mode = 'val_{}'.format(mode)
        if val_mode in history.history:
            plt.plot(range(epochs), history.history[val_mode][3:],
                     label='Validation {}'.format(label))
        plt.xlabel('Epoch')
        plt.ylabel(mode.capitalize())
        plt.legend(loc='upper right')

        self.save_plot('{}_history'.format(label))

    def plot_da_loss_history(self, history, mode='loss', label='Loss'):
        plt.figure()

        epochs = len(history.history[mode][3:])
        plt.plot(range(epochs), history.history[mode][3:], label=label)
        if 'r_loss' in history.history:
            plt.plot(range(epochs), history.history['r_loss'][3:],
                     label='Regression {}'.format(label))

        if 'da_loss' in history.history:
            plt.plot(range(epochs), history.history['da_loss'][3:],
                     label='Domain {}'.format(label))

        plt.xlabel('Epoch')
        plt.ylabel(mode.capitalize())
        plt.legend(loc='upper right')

        self.save_plot('{}_da_history'.format(label))

    def plot_vae_loss_history(self, history, mode='loss', label='Loss'):
        plt.figure()

        epochs = len(history.history[mode])
        plt.plot(range(epochs), history.history[mode], label=label)
        val_mode = 'val_{}'.format(mode)
        if val_mode in history.history:
            plt.plot(range(epochs), history.history[val_mode],
                     label='Validation {}'.format(label))
        if 'reg_loss_term' in history.history:
            plt.plot(range(epochs), history.history['reg_loss_term'],
                     label='Reg loss')
        if 'val_reg_loss_term' in history.history:
            plt.plot(range(epochs), history.history['val_reg_loss_term'],
                     label='Validation Reg loss')
        plt.xlabel('Epoch')
        plt.ylabel(mode.capitalize())
        plt.legend(loc='upper right')

        self.save_plot('{}_history'.format(label))

    def plot_flow_loss_history(self, history, mode='loss', label='Loss'):
        plt.figure()

        epochs = len(history.history[mode])
        plt.plot(range(epochs), history.history[mode], label=label)
        val_mode = 'val_{}'.format(mode)
        if val_mode in history.history:
            plt.plot(range(epochs), history.history[val_mode],
                     label='Validation {}'.format(label))
        plt.xlabel('Epoch')
        plt.ylabel(mode.capitalize())
        plt.legend(loc='upper right')

        self.save_plot('Flow_{}_history'.format(label))

    @staticmethod
    def scatter_predictions_vs_true(y_true, y_pred):
        plt.scatter(y_true, y_pred, color='b')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        xlims = [0, 1]
        ylims = [-0.2, 1.2]
        plt.xlim(xlims)
        plt.ylim(ylims)
        _ = plt.plot([0, 1], [0, 1])

    def plot_prediction_vs_true(self, y_true, y_pred):
        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred)
        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True')

    def plot_prediction_vs_true_with_error_bars(self, y_true, y_pred, err, correction=False):
        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred)
        plt.errorbar(y_true, y_pred, yerr=err, linestyle="None", fmt='o',
                     capsize=3, color='blue', capthick=0.5)

        # plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True_Error_Bars{}'.format('_corrected' if correction else ''))

    def plot_prediction_vs_true_with_error_bars_smooth(self, y_true, y_pred, err):
        sorted_idxs = np.argsort(y_true)
        y_true = y_true[sorted_idxs]
        y_pred = y_pred[sorted_idxs]
        err = np.array(err)[sorted_idxs]

        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred)

        plt.fill_between(y_true, y_pred + err, y_pred - err,
                         alpha=0.2, color='b')
        plt.fill_between(y_true, y_pred + 2 * err, y_pred - 2 * err,
                         alpha=0.2, color='b')

        # plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True_Error_Bars_Smooth')

    def plot_with_median(self, X, Y, err=None, color1='darkblue', color2=None,
                         label='Test', log=False, percentiles=True,
                         linestyle='--', fill_alpha=0.1, total_bins=15):
        """
        Plot the running media of the X, Y data with 16th and 84th percentiles,
        if requested
        """

        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        xlims = [0, 1]
        ylims = [-0.2, 1.2]
        plt.xlim(xlims)
        plt.ylim(ylims)
        _ = plt.plot([0, 1], [0, 1])

        if color2 is None:
            color2 = color1

        if log:
            bins = np.geomspace(X.min(), X.max(), total_bins)
        else:
            bins = np.linspace(X.min(), X.max(), total_bins)

        delta = bins[1] - bins[0]
        idx = np.digitize(X, bins)
        running_median = [np.median(Y[idx == k]) for k in range(total_bins)]
        running_prc25 = [np.nanpercentile(Y[idx == k], 16) for k in range(total_bins)]
        running_prc75 = [np.nanpercentile(Y[idx == k], 84) for k in range(total_bins)]

        if percentiles:
            plt.plot(bins - delta / 2, running_median, color1, linestyle=linestyle, lw=2, alpha=.8, label=label)
            plt.fill_between(bins - delta / 2, running_prc25, running_median, facecolor=color2, alpha=fill_alpha)
            plt.fill_between(bins - delta / 2, running_prc75, running_median, facecolor=color2, alpha=fill_alpha)
        if err is not None:
            plt.errorbar(X, Y, yerr=err, linestyle="None", fmt='o',
                         capsize=3, color='blue', capthick=0.5)

            # ax.set_aspect('equal')
        # else:
        #    ax.plot(bins - delta / 2, running_median, color1, linestyle='--', lw=2, alpha=.8,  label=label)

        if log:
            plt.xscale('symlog')

        self.save_plot('Predictions_vs_True_percentiles')

    def calculate_corrected_y_pred(self, y_true, y_pred_distr, prior):
        x = np.linspace(0.01, 0.99, 50)
        xt = x.reshape((-1, 1))

        logps = []
        for i in range(len(x)):
            logps.append(y_pred_distr.log_prob(xt[i]).numpy())

        logps = np.stack(logps)

        for i in range(10):
            plt.figure()
            plt.plot(x, np.exp(logps[:, -i]), label='posterior under training prior')
            plt.plot(x, np.exp(logps[:, -i]) / prior.pdf(x), label='posterior under flat prior')
            plt.axvline(y_true[-i], color='m', label='True value')
            plt.xlabel('True Values')
            plt.legend(loc='upper left')
            self.save_plot('Distr_{}'.format(i))

        corrected_posterior = np.exp(logps) / (prior.pdf(x).reshape((-1, 1)))
        corrected_posterior[corrected_posterior <= 0] = 1e-2

        y_pred_prior_mean = (simps(x.reshape((-1, 1)) * corrected_posterior, x, axis=0) /
                             simps(corrected_posterior, x, axis=0))

        fig, ax = plt.subplots(5, 8, sharex='col', sharey='row', figsize=(20, 10))
        for i in range(5):
            for j in range(8):
                ax[i, j].axvline(y_true[10 * i + j], label='Ground Truth', color='m')
                ax[i, j].axvline(x[np.argmax(np.exp(logps[:, 10 * i + j]))], label='Predicted Value', color='darkblue')
                ax[i, j].axvline(y_pred_prior_mean[10 * i + j], label='Corrected Predicted Value', color='darkgreen')
                ax[i, j].plot(x, np.exp(logps[:, 10 * i + j]), label='Posterior under training prior', color='blue')
                ax[i, j].plot(x, np.exp(logps[:, 10 * i + j]) / prior.pdf(x), label='Posterior under flat prior',
                              color='green')
                # ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), ncol=3, loc='upper center', fancybox=True, shadow=True)
        plt.tight_layout()
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Exsitu stellar mass Fraction')
        self.save_plot('Distr_all')

        corrected_posterior = np.exp(logps) / (prior.pdf(x).reshape((-1, 1)))
        corrected_posterior[corrected_posterior <= 0] = 1e-2

        y_pred_prior_mean = (simps(x.reshape((-1, 1)) * corrected_posterior, x, axis=0) /
                             simps(corrected_posterior, x, axis=0))

        y_pred_mode = x[np.exp(corrected_posterior).argmax(axis=0)]
        return y_pred_prior_mean, y_pred_mode

    def plot_with_percentiles(self, y_true, y_pred, corrected=0):
        f = plt.figure(figsize=(6, 8))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

        ax1 = f.add_subplot(gs[0, 0])

        plt.plot(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1), 'k--')
        _ = self.binned_plot(y_true,
                             y_pred,
                             n=20, percentiles=[35, 45, 50],
                             color='b', ax=ax1)

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xticks([])
        ax1.set_yticks(ax1.get_yticks()[1:])
        ax1.set_ylabel('Predictions', fontsize=14)
        ax1.legend(loc='upper left')

        ax2 = f.add_subplot(gs[1, 0])

        ax2.plot(np.arange(0, 1, 0.1), [0] * 10, 'k--')
        _ = self.binned_plot(y_true,
                             y_pred - y_true,
                             n=20, percentiles=[35, 45, 50],
                             color='b', ax=ax2)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('True Values', fontsize=14)
        ax2.set_ylabel(r'$\epsilon$', fontsize=14)

        self.save_plot('Predictions_vs_True_Percentiles_{}'.format(corrected))

    def plot_saliency_maps(self, model, images, y_true, y_pred,
                           m_steps=240, cmap="jet", overlay_alpha=0.4):

        baseline = tf.zeros(shape=(128, 128, 5))
        for i in range(15):
            image = images[i]
            attributions = integrated_gradients(model=model,
                                                baseline=baseline,
                                                image=image,
                                                m_steps=m_steps)

            fig, big_axes = plt.subplots(figsize=(12.0, 12.0), nrows=5, ncols=1, sharey=True)

            for channel, big_ax in enumerate(big_axes):
                big_ax.set_title('Channel: {}'.format(channel), rotation=0)
                # Turn off axis lines and ticks of the big subplot
                # obs alpha is 0 in RGBA string!
                big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
                # removes the white frame
                big_ax._frameon = False

                attribution_mask = attributions[:, :, channel].numpy()

                ax = fig.add_subplot(5, 3, 3 * channel + 1)
                # ax.set_title('Original image')
                col = ax.pcolor(image[:, :, channel], cmap="jet")
                fig.colorbar(col, ax=ax, fraction=0.046, pad=0.04)
                # ax.axis('off')

                ax = fig.add_subplot(5, 3, 3 * channel + 2)
                # ax.set_title('Attribution mask')

                vmin = attribution_mask.min()
                vmax = attribution_mask.max()
                if vmin >= 0 or vmax <= 0:
                    norm = None
                else:
                    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

                col = plt.pcolormesh(attribution_mask, norm=norm, cmap="seismic")

                # col = ax.imshow(attribution_mask, cmap=cmap)
                # ax.axis('off')
                # fig.colorbar(col, ax=ax, fraction=0.046, pad=0.04)

                ax = fig.add_subplot(5, 3, 3 * channel + 3)
                # ax.set_title('Overlay')
                # col = ax.imshow(attribution_mask, cmap=cmap)
                col = plt.pcolormesh(attribution_mask, norm=norm, cmap="seismic")
                ax.pcolormesh(image[:, :, channel], alpha=overlay_alpha, cmap=plt.cm.gray)
                # ax.axis('off')
                fig.colorbar(col, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle('Saliency maps for Galaxy with Exsitu F {:.2f} '
                         'and Predicted {:.2f}'.format(y_true[i], y_pred[i]))
            plt.tight_layout()
            self.save_plot('Saliency_maps_{}'.format(i), directory='Saliency_Maps')

    def plot_feature_maps(self, model, images, y_true, y_pred):

        # redefine model to output right after the first hidden layer
        ixs = [1, 4, 6]
        outputs = [model.layers[i].output for i in ixs]
        model = Model(inputs=model.inputs, outputs=outputs)

        for i in range(15):
            # expand dimensions so that it represents a single 'sample'
            img = np.expand_dims(images[i], axis=0)

            # get feature map for first hidden layer
            feature_maps = model.predict(img)

            # plot the output from each block
            square = 4
            for fcount, fmap in enumerate(feature_maps):
                # plot all 64 maps in an 8x8 squares
                ix = 1
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        ax = plt.subplot(square, square, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel in grayscale
                        plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                        ix += 1

                # show the figure
                plt.suptitle('Feature map {} for Galaxy with Insitu {:.2f} '
                             'and Predicted {:.2f}'.format(fcount, y_true[i], y_pred[i]))
                plt.tight_layout()
                self.save_plot('Feature_Maps_{}_{}'.format(fcount, i), directory='Feature_Maps')

    @staticmethod
    def binned_plot(X, Y, n=10, percentiles=[35, 50], ax=None, **kwargs):
        # Calculation
        calc_percent = []
        for p in percentiles:
            if p < 50:
                calc_percent.append(50 - p)
                calc_percent.append(50 + p)
            elif p == 50:
                calc_percent.append(50)
            else:
                raise Exception('Percentile > 50')

        bin_edges = np.linspace(X.min() * 0.9999, X.max() * 1.0001, n + 1)

        dtype = [(str(i), 'f') for i in calc_percent]
        bin_data = np.zeros(shape=(n,), dtype=dtype)

        for i in range(n):
            y = Y[(X >= bin_edges[i]) & (X < bin_edges[i + 1])]

            if len(y) == 0:
                continue

            y_p = np.percentile(y, calc_percent)

            bin_data[i] = tuple(y_p)

        # Plotting
        if ax is None:
            f, ax = plt.subplots()

        bin_centers = [np.mean(bin_edges[i:i + 2]) for i in range(n)]
        for p in percentiles:
            if p == 50:
                ax.plot(bin_centers, bin_data[str(p)], label='50-percentile', **kwargs)
            else:
                ax.fill_between(bin_centers,
                                bin_data[str(50 - p)],
                                bin_data[str(50 + p)],
                                alpha=0.2,
                                label='{}-percentile'.format(p),
                                **kwargs)

        return bin_data, bin_edges



