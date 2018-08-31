import os
import re
import sys

from collections import OrderedDict

def check_out_of_memory(log_file):
    check_str = "Check failed: error == cudaSuccess"
    check_str2 = "SIGSEGV"
    with open(log_file, 'r') as f:
        for line in f:
            if check_str in line or check_str2 in line:
                print "Caffe out of memory detected!"
                return True
    return False

def net_string_to_id(net_string):
    return net_string.replace('(', '').replace(')', '').replace(' ', '').replace(',', '-')

# helper function to parse log file.
def parse_line_for_net_output(regex_obj, row, row_dict_list, line, iteration, model_id=None, logger=None, test=False):
    output_match = regex_obj.search(line)
    if output_match:
        if not row or row['NumIters'] != iteration:
            # Push the last row and start a new one.
            if row:
                row_dict_list.append(row)
            row = {'NumIters':  iteration}

        # Get the key value pairs from a line.
        output_name = output_match.group(3)
        output_val = output_match.group(4)
        output_time = output_match.group(1)
        row[output_name] = float(output_val)
        
        # we want to log these values.
        if test and logger:
            if (output_name == 'Accuracy1'):
                log_output_name = 'val_acc'
            elif (output_name == 'SoftmaxWithLoss1'):
                log_output_name = 'loss'
            else:
                log_output_name = output_name

            logger.log_measurements({'id': net_string_to_id(model_id)}, iteration, 
                {log_output_name: output_val }, '2018-08-07T' + output_time
            )

        # row['timestamp'] = output_time
    # Check if this row is the last for this dictionary.
    if row and len(row_dict_list) >= 1 and len(row) == len(row_dict_list[0]):
    # actually, we want to include all measurements, for visualization sake
    # if row:
        row_dict_list.append(row)
        row = None
    return row_dict_list, row

# I0806 11:11:15.483248 14351 solver.cpp:239] Iteration 200 (43.9273 iter/s, 2.27649s/100 iters), loss = 1.74986
# I0806 11:11:15.483296 14351 solver.cpp:258]     Train net output #0: Accuracy1 = 0.375
# I0806 11:11:15.483306 14351 solver.cpp:258]     Train net output #1: SoftmaxWithLoss1 = 1.74986 (* 1 = 1.74986 loss)
# I0806 11:11:15.483327 14351 sgd_solver.cpp:112] Iteration 200, lr = 0.001
# I0806 11:11:17.759376 14351 solver.cpp:239] Iteration 300 (43.9377 iter/s, 2.27595s/100 iters), loss = 1.75921
# I0806 11:11:17.759420 14351 solver.cpp:258]     Train net output #0: Accuracy1 = 0.375
# I0806 11:11:17.759434 14351 solver.cpp:258]     Train net output #1: SoftmaxWithLoss1 = 1.75921 (* 1 = 1.75921 loss)
# I0806 11:11:17.759443 14351 sgd_solver.cpp:112] Iteration 300, lr = 0.001
# I0806 11:11:20.034107 14351 solver.cpp:239] Iteration 400 (43.9646 iter/s, 2.27456s/100 iters), loss = 1.65272
# I0806 11:11:20.034164 14351 solver.cpp:258]     Train net output #0: Accuracy1 = 0.375
# I0806 11:11:20.034185 14351 solver.cpp:258]     Train net output #1: SoftmaxWithLoss1 = 1.65272 (* 1 = 1.65272 loss)
# I0806 11:11:20.034195 14351 sgd_solver.cpp:112] Iteration 400, lr = 0.001
# I0806 11:11:22.318338 14351 solver.cpp:239] Iteration 500 (43.7818 iter/s, 2.28405s/100 iters), loss = 1.68612

# MAIN FUNCTION: parses log file.
# UPDATE: want to parse datetime, as well as loss and accuracy
def parse_caffe_log_file(log_file, model_id=None, logger=None):
    print "Parsing [%s]" % log_file
    regex_iteration = re.compile('Iteration (\d+)')
    regex_train_output = re.compile(
            '(Train) net output #(\d+): (\S+) = ([\.\deE+-]+)')
    regex_test_output = re.compile(
            '(\d{2}:\d{2}:\d{2}.\d{6}).*Test net output #(\d+): (\S+) = ([\.\deE+-]+)')
    iteration = -1
    train_dict_list = []
    test_dict_list = []
    snapshot_list = []
    train_row = None
    test_row = None

    with open(log_file, 'r') as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                continue
            # Parse for test/train accuracies and loss.
            train_dict_list, train_row = parse_line_for_net_output(
                    regex_train_output, train_row, train_dict_list, line, iteration, model_id, logger)
            test_dict_list, test_row = parse_line_for_net_output(
                    regex_test_output, test_row, test_dict_list, line, iteration, model_id, logger, test=True)
 
    if test_dict_list == [] and test_row:
        test_dict_list = [test_row]

    return train_dict_list, test_dict_list


# Gets all accuracies as a list.
def get_all_accuracies(log_file):
    test_acc_list = []
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    for test_acc in test_dict_list:
        test_acc_list.append(test_acc['Accuracy1'])
    return test_acc_list


# Helper function to get accuracies as dict.
def get_test_accuracies_dict(log_file):
    test_acc_dict = OrderedDict()
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    for test_acc in test_dict_list:
        test_acc_dict[int(test_acc['NumIters'])] = test_acc['Accuracy1']
    return test_acc_dict


def get_snapshot_list(log_file):
    print "Parsing [%s] for snapshots" % log_file
    regex_iteration = re.compile('Iteration (\d+)')
    regex_snapshot = re.compile('Snapshotting solver state to binary proto file (\S+)')
    iteration = -1
    snapshot_list = []

    with open(log_file, 'r') as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                iteration = float(iteration_match.group(1))
            if iteration == -1:
                # Only start parsing for other stuff if we've found the first
                # iteration
                continue
            snapshot_match = regex_snapshot.search(line)
            if snapshot_match:
                snapshot_file = snapshot_match.group(1)
                iter_no = int(snapshot_file.split(".")[0].split("_")[-1])
                snapshot_list.append((iter_no, snapshot_file))
    
    return snapshot_list

# Helper function to get epoch.
def get_last_epoch_snapshot(log_file):
    snapshot_list = get_snapshot_list(log_file)
    print snapshot_list
    last_iter, snapshot_file = snapshot_list[-1]
    return last_iter, snapshot_file

# Helper function to get epoch.
def get_last_test_epoch(log_file):
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file)
    if len(test_dict_list) == 0:
        return -1, []
    last_test_epoch = test_dict_list[-1]
    return last_test_epoch['NumIters'], last_test_epoch

# Run caffe command line and return accuracies.
def run_caffe_return_accuracy(solver_fname, log_file, caffe_root, num_iter=-1, gpu_to_use=None, model_id=None, logger=None):
    cmd_suffix = ''
    if num_iter > 0:
        cmd_suffix = ' --iterations %d ' % num_iter

    if gpu_to_use is not None:
        run_cmd = '%s train --solver %s --gpu %i %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, gpu_to_use, cmd_suffix, log_file)
    else:
        run_cmd = '%s train --solver %s %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, cmd_suffix, log_file)

    # Run the caffe code.
    print "Running [%s]" % run_cmd
    os.system(run_cmd)

    # Get the accuracy values.
    if check_out_of_memory(log_file):
        return None, None
    train_dict_list, test_dict_list = parse_caffe_log_file(log_file, model_id=model_id, logger=logger)

    acc = test_dict_list[-1]['Accuracy1']
    acc_dict = {test_dict_list[-1]['NumIters']: test_dict_list[-1]['Accuracy1']}
    return acc, acc_dict

def run_caffe_from_snapshot(solver_fname, log_file, snapshot_file, caffe_root, gpu_to_use=None):
    if gpu_to_use is not None:
        run_cmd = '%s train --solver %s --gpu %i --snapshot %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, gpu_to_use, snapshot_file, log_file)
    else:
        run_cmd = '%s train --solver %s --snapshot %s >> %s 2>&1 ' % (
                os.path.join(caffe_root, 'build/tools/caffe'), solver_fname, snapshot_file, log_file)

    # Run the caffe code.
    print "Running [%s]" % run_cmd
    os.system(run_cmd)

    test_acc_list = get_all_accuracies(log_file)
    return test_acc_list