from sklearn.metrics import confusion_matrix
import sys
sys.path.append('.')
import numpy as np

import torch
import matplotlib.pyplot as plt



from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, precision_recall_curve,auc,accuracy_score
from sklearn.metrics import average_precision_score


import warnings

warnings.filterwarnings('ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ROCAUCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):

        self.y_true_11=None
        self.y_pred_11 = None

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()

        y_pred = torch.sigmoid(y_pred).data.cpu().numpy()

        if self.y_true_11 is None:
            self.y_true_11 = y_true
            self.y_pred_11 = y_pred
        else:
            self.y_true_11 = np.concatenate((self.y_true_11, y_true),axis=0)
            self.y_pred_11 = np.concatenate((self.y_pred_11, y_pred),axis=0)

        return self.y_true_11, self.y_pred_11

    def fast_auc(self,y_true, y_prob):


        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        cumfalses = np.cumsum(1 - y_true)
        nfalse = cumfalses[-1]
        auc = (y_true * cumfalses).sum()

        auc /= (nfalse * (len(y_true) - nfalse))
        return auc

    @property
    def avg(self):

        self.y_true_11=self.y_true_11.reshape(-1)
        self.y_pred_11 = self.y_pred_11.reshape(-1)
        score=self.fast_auc(self.y_true_11, self.y_pred_11)

        return score

    def report_with_recall_precision(self, y_true_1, y_pre_1):

        '''
        precisions, recalls, thresholds = precision_recall_curve(y_true_1, y_pre_1)
        youden = precisions + recalls
        cutoff = thresholds[np.argmax(youden)]
        y_pred_t = [1 if i > cutoff else 0 for i in y_pre_1]
        tn, fp, fn, tp = confusion_matrix(y_true_1, y_pred_t).ravel()
        recall = round(recall_score(y_true_1, y_pred_t), 4)
        precision = round(precision_score(y_true_1, y_pred_t), 4)
        f1 = round(f1_score(y_true_1, y_pred_t), 4)
        print('for recall: %.4f, precision: %.4f,tn: %d,fp: %d,fn: %d,tp: %d,f1: %.4f, cutoff: %.8f'
              % (recall, precision, tn, fp, fn, tp, f1, cutoff))
        '''

        precisions, recalls, thresholds = precision_recall_curve(y_true_1, y_pre_1)

        y_true_111 = y_true_1.reshape(-1)
        y_pred_111 = y_pre_1.reshape(-1)

        for i in range(1, 20):
            r = i / 20

            for j in range(len(recalls)):
                try:
                    if float(recalls[j]) < float(r):
                        recall = round(recalls[j - 1], 4)
                        #threshold = round(thresholds[j - 1], 4)
                        threshold = thresholds[j - 1]
                        #print("threshold:", threshold)
                        y_pre = y_pred_111 > threshold
                        tn, fp, fn, tp = confusion_matrix(y_true_111, y_pre).ravel()
                        precision = precision_score(y_true_111, y_pre)
                        recall = recall_score(y_true_111, y_pre)
                        f1 = f1_score(y_true_111, y_pre)
                        print('for recall: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f,f1: %.4f, threshold: %.8f' % (recall, tn, fp, fn, tp, precision, f1, threshold))

                        break
                except Exception as e:
                    print("=====e====", e)

    def report_tpr_acc(self, video_y_true, video_y_pre, base_y_true, base_y_pre, img_path):
        # 准确率
        video_accuracy = accuracy_score(video_y_true, video_y_pre)
        base_accuracy = accuracy_score(base_y_true, base_y_pre)
        # 特异性
        video_fpr, video_tpr, video_thresholds = roc_curve(video_y_true, video_y_pre)
        base_fpr, base_tpr, base_thresholds = roc_curve(base_y_true, base_y_pre)

        lines = []
        labels = []

        l, = plt.plot(video_accuracy, video_tpr, color='navy', lw=2)  # 划线
        lines.append(l)
        labels.append('video_accuracy_sensitivity')

        l, = plt.plot(base_accuracy, base_tpr, color='darkorange', lw=2)  # 划线
        lines.append(l)
        labels.append('base_accuracy_sensitivity')

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0, 1.1])
        plt.xticks(np.linspace(0, 1.0, 10))
        plt.ylim([0, 1.1])
        plt.yticks(np.linspace(0, 1.0, 10))
        plt.xlabel('accuracy')
        plt.ylabel('sensitivity')
        plt.title(
            'accuracy sensitivity compare image')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        plt.savefig(img_path, dpi=300, bbox_inches='tight')

    def report_tpr_fpr(self, video_y_true, video_y_pre, base_y_true, base_y_pre, img_path):
        # 特异性
        video_fpr, video_tpr, video_thresholds = roc_curve(video_y_true, video_y_pre)
        base_fpr, base_tpr, base_thresholds = roc_curve(base_y_true, base_y_pre)
        video_auc2 = auc(video_fpr, video_tpr)
        base_auc2 = auc(base_fpr, base_tpr)
        print('video  auc : {0:0.2f}'.format(
            video_auc2))
        print('base auc : {0:0.2f}'.format(
            base_auc2))

        lines = []
        labels = []


        fig11, ax = plt.subplots()
        # 将右侧和上方的边框线设置为不显示
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        l, = ax.plot(video_fpr, video_tpr, color='navy', lw=2)  # 划线
        lines.append(l)
        labels.append('video_specificity_sensitivity (auc = {0:0.2f})'
                      ''.format(video_auc2))

        l, = ax.plot(base_fpr, base_tpr, color='darkorange', lw=2)  # 划线
        lines.append(l)
        labels.append('base_specificity_sensitivity (auc = {0:0.2f})'
                      ''.format(base_auc2))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)

        ax.set_xlim([0, 1.0])
        ax.set_xticks(np.linspace(0, 1.0, 11))
        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.linspace(0, 1.0, 11))
        font = {'family': 'times new roman', 'size': 14}
        ax.set_xlabel('1-specificity', fontdict=font)
        ax.set_ylabel('sensitivity', fontdict=font)
        ax.set_title('specificity sensitivity compare image', fontdict=font)
        plt.legend(lines, labels, loc=(0, -.38), prop=font)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')

    def report_with_recall(self, video_y_true, video_y_pre, base_y_true, base_y_pre, root_path):

        video_precision, video_recall, video_thresholds = precision_recall_curve(video_y_true, video_y_pre)
        base_precision, base_recall, base_thresholds = precision_recall_curve(base_y_true, base_y_pre)

        video_youden = video_precision + video_recall
        video_thresholds_cutoff = video_thresholds[np.argmax(video_youden)]
        video_precision_cutoff = video_precision[np.argmax(video_youden)]
        video_recall_cutoff = video_recall[np.argmax(video_youden)]
        print("======precision_recall video_thresholds_cutoff ====", video_thresholds_cutoff)
        print("======precision_recall video_precision_cutoff ====", video_precision_cutoff)
        print("======precision_recall video_recall_cutoff ====", video_recall_cutoff)

        base_youden = base_precision + base_recall
        base_thresholds_cutoff = base_thresholds[np.argmax(base_youden)]
        base_precision_cutoff = base_precision[np.argmax(base_youden)]
        base_recall_cutoff = base_recall[np.argmax(base_youden)]
        print("======precision_recall base_thresholds_cutoff ====", base_thresholds_cutoff)
        print("======precision_recall base_precision_cutoff ====", base_precision_cutoff)
        print("======precision_recall base_recall_cutoff ====", base_recall_cutoff)


        np.savetxt(root_path + '_video_precision.txt', video_precision, fmt='%f')
        np.savetxt(root_path + '_video_recall.txt', video_recall, fmt='%f')
        np.savetxt(root_path + '_video_thresholds_pr.txt', video_thresholds, fmt='%f')
        np.savetxt(root_path + '_base_precision.txt', base_precision, fmt='%f')
        np.savetxt(root_path + '_base_recall.txt', base_recall, fmt='%f')
        np.savetxt(root_path + '_base_thresholds_pr.txt', base_thresholds, fmt='%f')


        video_average_precision = average_precision_score(video_y_true, video_y_pre)
        base_average_precision = average_precision_score(base_y_true, base_y_pre)
        print('video  precision-recall score: {0:0.4f}'.format(
            video_average_precision))
        print('base precision-recall score: {0:0.4f}'.format(
            base_average_precision))
        # print("precision:", precision, "recall:", recall, "thresholds:", thresholds)

        lines = []
        labels = []

        fig11, ax = plt.subplots()
        # 将右侧和上方的边框线设置为不显示
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        l, = plt.plot(video_recall, video_precision, color='navy', lw=2)
        lines.append(l)
        labels.append('add-video Precision-recall (area = {0:0.4f})'
                      ''.format(video_average_precision))

        l, = plt.plot(base_recall, base_precision, color='darkorange', lw=2)
        lines.append(l)
        labels.append('base-line Precision-recall (area = {0:0.4f})'
                      ''.format(base_average_precision))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)

        ax.set_xlim([0, 1.0])
        ax.set_xticks(np.linspace(0, 1.0, 11))
        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.linspace(0, 1.0, 11))
        font = {'family': 'times new roman', 'size': 14}
        ax.set_xlabel('Recall', fontdict=font)
        ax.set_ylabel('Precision', fontdict=font)
        ax.set_title('sprecision recall compare image', fontdict=font)
        plt.legend(lines, labels, loc=(0, -.38), prop=font)

        img_path = root_path.split(".")[0] + "_Precision_Recall.jpg"

        plt.savefig(img_path, dpi=300, bbox_inches='tight')

    def report_return_08PR(self):

        precisions, recalls, thresholds = precision_recall_curve(self.y_true_11, self.y_pred_11)
        # pdb.set_trace()
        y_true_111 = self.y_true_11.reshape(-1)
        y_pred_111 = self.y_pred_11.reshape(-1)

        for i in range(1, 21):
            r = i / 20

            for j in range(len(recalls)):
                if float(recalls[j]) < float(r):
                    recall = round(recalls[j - 1], 4)
                    # threshold = round(thresholds[j - 1], 4)
                    threshold = thresholds[j - 1]
                    # print("threshold:", threshold)
                    # precision = round(precisions[j - 1], 4)
                    y_pre = y_pred_111 > threshold
                    # pdb.set_trace()  # Set breakpoint
                    # tn, fp, fn, tp = confusion_matrix(y_true_111, y_pre).ravel()
                    tp = np.sum((y_true_111 == 1) & (y_pre == 1))
                    fp = np.sum((y_true_111 == 0) & (y_pre == 1))
                    tn = np.sum((y_true_111 == 0) & (y_pre == 0))
                    fn = np.sum((y_true_111 == 1) & (y_pre == 0))
                    precision = precision_score(y_true_111, y_pre)
                    recall = recall_score(y_true_111, y_pre)
                    f1 = f1_score(y_true_111, y_pre)
                    print('for recall: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f,f1: %.4f, threshold: %.8f' % (
                        recall, tn, fp, fn, tp, precision, f1, threshold))
                    if i == 16:
                        PR_back = precision
                    break
        return PR_back

    def evaluate(y_true, y_pred, digits=4, cutoff='auto'):

        if cutoff == 'auto':
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            youden = tpr - fpr
            cutoff = thresholds[np.argmax(youden)]

        return cutoff

    def report(self):

        self.y_true_11 = self.y_true_11.reshape(-1)
        self.y_pred_11 = self.y_pred_11.reshape(-1)

        for score in range(1, 20):
            score = score / 20
            y_pre = self.y_pred_11 > score

            tn, fp, fn, tp = confusion_matrix(self.y_true_11, y_pre).ravel()

            precision = precision_score(self.y_true_11, y_pre)
            recall = recall_score(self.y_true_11, y_pre)
            f1 = f1_score(self.y_true_11, y_pre)

            print('for threshold: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f, '
                  'recall: %.4f, f1: %.4f' % (score, tn, fp, fn, tp, precision, recall, f1))

            # print('for threshold: %.4f, precision: %.4f, recall: %.4f, f1: %.4f' % (score, precision, recall, f1))
        for score in range(90, 100):
            score = score / 100
            y_pre = self.y_pred_11 > score

            tn, fp, fn, tp = confusion_matrix(self.y_true_11, y_pre).ravel()

            precision = precision_score(self.y_true_11, y_pre)
            recall = recall_score(self.y_true_11, y_pre)
            f1 = f1_score(self.y_true_11, y_pre)

            print('for threshold: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f, '
                  'recall: %.4f, f1: %.4f' % (score, tn, fp, fn, tp, precision, recall, f1))

        return score


    '''
    def report_for_list(self,y_true_1, y_pre_1):


        for score in range(1, 20):

            y_true_111 = y_true_1.reshape(-1)
            y_pred_111 = y_pre_1.reshape(-1)

            score = score / 20
            y_pre = y_pred_111 > score

            tn, fp, fn, tp = confusion_matrix(y_true_1, y_pre_1).ravel()


            precision = precision_score(y_true_111, y_pre)
            recall = recall_score(y_true_111, y_pre)
            f1 = f1_score(y_true_111, y_pre)

            print('for threshold: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f, '
                  'recall: %.4f, f1: %.4f' % (score, tn, fp, fn, tp, precision, recall, f1))

            # print('for threshold: %.4f, precision: %.4f, recall: %.4f, f1: %.4f' % (score, precision, recall, f1))
        for score in range(90, 100):
            score = score / 100
            y_pre = self.y_pred_11 > score

            tn, fp, fn, tp = confusion_matrix(self.y_true_11, y_pre).ravel()

            precision = precision_score(self.y_true_11, y_pre)
            recall = recall_score(self.y_true_11, y_pre)
            f1 = f1_score(self.y_true_11, y_pre)

            print('for threshold: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f, '
                  'recall: %.4f, f1: %.4f' % (score, tn, fp, fn, tp, precision, recall, f1))

        return score
    '''

    def save_label_score(self, path):

        self.y_true_11 = self.y_true_11.reshape(-1)
        self.y_pred_11 = self.y_pred_11.reshape(-1)
        labels = self.y_true_11.tolist()
        y_pres = self.y_pred_11.tolist()

        print("labels length:", len(labels))
        print("y_pres length:", len(y_pres))



        with open(path, 'w') as f:
            for x, y in zip(labels, y_pres):
                f.write(str(x) + ',' + str(y) + '\n')

    def save_score_with_recall(self, score):

        print("==========save_score_with_recall============")

        precision, recall, thresholds = precision_recall_curve(self.y_true_11,  self.y_pred_11)

        for i in range(len(recall)):
            if float(recall[i]) == float(score):

                score = thresholds[1]
                y_pre = self.y_pred_11 > score

                tn, fp, fn, tp = confusion_matrix(self.y_true_11, y_pre).ravel()

                print('for threshold: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f, '
                      'recall: %.4f' % (score, tn, fp, fn, tp, precision, recall))


if __name__=='__main__':
    ROCAUC_score = ROCAUCMeter()

    y_true = np.random.randint(2, size=10000)
    y_prob = np.random.rand(10000)

    ROCAUC_score.update(y_true,y_prob)
    print(ROCAUC_score.avg)
