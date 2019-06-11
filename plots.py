from visdom import Visdom
import numpy as np
#from configs import index2name, name2index
import plotly.graph_objs as go
import os
import json

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', plot_path='/tmp'):
        self.viz = Visdom(port=6065)
        self.env = env_name
        self.plots = {}
        self.plot_path = plot_path
        #self.viz.log_to_filename = log_filename

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def save_plots(self):
        windowstring = self.viz.get_window_data(env=self.env)
        windowdict = json.loads(windowstring)
        for windowname, windowcontent in windowdict.items():
            data  = [{k:datapoint[k] for k in ['x','y','type','name', 'mode']} for datapoint in windowcontent['content']['data']]
            layout = windowcontent['content']['layout']
            plotlyfig = dict(data=data, layout=layout)
            plot_path = os.path.join(self.plot_path,windowcontent['title']+'.html')
            print("Saving plot to:", plot_path)
            plot(plotlyfig, filename=os.path.join(self.plot_path,windowcontent['title']+'.html'), auto_open=False)



def plot_epoch_end(plotter, phase, epoch, epoch_acc, epoch_loss, lr, running_class_stats):
    """ Plot statistics to visdom """
    plotter.plot('acc', phase, 'acc', epoch, epoch_acc)
    plotter.plot(var_name='loss', split_name=phase, title_name='loss', x=epoch, y=epoch_loss)
    plotter.plot(var_name='LR', split_name='LR', title_name='LR', x=epoch, y=lr)
    plotter.plot(var_name='LR', split_name='LR', title_name='LR', x=epoch, y=lr)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    # for key, val in class_acc.items():
    #     #print(key, val)
    #     classname = dataset_val.index2name[key]
    for classname in running_class_stats.keys():
        TP = running_class_stats[classname]['TP']
        TN = running_class_stats[classname]['TN']
        FP = running_class_stats[classname]['FP']
        FN = running_class_stats[classname]['FN']
        class_acc = round(float((TP + TN) / (TP + TN + FP + FN + 1e-3)), 2)
        class_prec = round(float(TP / (TP + FP + 1e-3)), 2)
        class_recall = round(float(TP / (TP + FN + 1e-3)), 2)
        print(TP, TN, FP, FN, 'ooo')
        print(class_acc, class_prec, class_recall, "---")

        # plotter_acc.plot(var_name=key, split_name=phase, title_name='class_acc', x=epoch, y=val)
        plotter.plot(var_name='acc_' + phase, split_name=classname, title_name='class_acc_' + phase, x=epoch,
                     y=class_acc)
        plotter.plot(var_name='prec_' + phase, split_name=classname, title_name='class_prec_' + phase, x=epoch,
                     y=class_prec)
        plotter.plot(var_name='recall_' + phase, split_name=classname, title_name='class_recall_' + phase, x=epoch,
                     y=class_recall)
        plotter.plot(var_name='num_preds_' + phase, split_name=classname,
                     title_name='num_preds_' + phase, x=epoch, y=running_class_stats[classname]['num_preds'])
        plotter.plot(var_name='num_gt_' + phase, split_name=classname,
                     title_name='num_gt_' + phase, x=epoch, y=running_class_stats[classname]['num_gt'])
    return


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def bar_chart(val_annotations, outfile):
    classnames = set(val_annotations.pred_classname)

    gt2pred_abs = pd.DataFrame(index=classnames, columns=classnames)
    gt2pred_rel = pd.DataFrame(index=classnames, columns=classnames)


    for gt_classname in classnames:
        for pred_classname in classnames:
            gt2pred_abs.loc[gt_classname][pred_classname] = sum((val_annotations.classname == gt_classname) &
                                                                (val_annotations.pred_classname == pred_classname))
    for pred_classname in classnames:
        gt2pred_rel[pred_classname] = gt2pred_abs[pred_classname] / sum(gt2pred_abs[pred_classname])

    data = []

    for classname in classnames:
        trace = go.Bar(
            x=gt2pred_rel.columns,
            y=gt2pred_rel.loc[classname],
            # text=gt2pred_abs[classname],
            textposition='auto',
            name=classname
        )
        data.append(trace)

    layout = go.Layout(
        barmode='stack',
        title='Prediction statistics',
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename=outfile, auto_open=False)
