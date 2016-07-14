"""
@author: dncampo
"""

import os
from index import fm, oc, jac
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from youtube_experiment import *

max_id_possible = -1
data_path = '../datasets/'
data_input = 'com-youtube.all16386.txt'
dataset_file = data_path + data_input

fp = open(dataset_file, 'r')
exp_reps = 20
num_of_users = 10  # number of minimum users per comm
lines = fp.readlines()
fp.close()

# Processing the dataset
# cut out the communities with less than 'num_of_users' users
new_dataset = [x for x in lines if len(x.split()) >= num_of_users]
# convert ID's to int
new_dataset = [[int(y) for y in xx.split()] for xx in new_dataset]

# the instance experiment
an_exp = Youtube_experiments()

# make IDs consecutive
an_exp.create_consecutives_ids(new_dataset)

# count the overlap of each comm
comm_overlap = an_exp.get_overlaps()

# sort by degree of overlap
sorted_overlaps = an_exp.get_sorted_overlaps

# build the  bmus from a dictionary
# key = level of overlap
# value = list of communities with same degree as the key
# each comm istself is a list of IDs
bmus, overlap_bmus = an_exp.create_bmus()


# EXPERIMENTS
considered_overlaps = numpy.unique(overlap_bmus)

fs = [0.1]  # percentage of perturbation
dir_figs = "figs_" + data_input + "_min_num_users_" + str(num_of_users)
results_fm = []
results_oc = []
results_jac = []
results_path = "../results/" + dir_figs + "/"

if not os.path.exists(results_path):
    os.makedirs(results_path)

pdfname = "results.pdf"
pp = PdfPages(results_path + pdfname)

for f in fs:
    fm_slice = []
    oc_slice = []
    jac_slice = []
    avg_overlap = []
    avg_overlap_S2 = []
    print "f %.2f: " % f

    # divide the dataset into set of communities with similar degree of overlap
    new_bases, new_considered_overlaps = an_exp.break_down_overlaps(considered_overlaps)

    for oo in range(1, len(new_bases)):

        # indexes of desired overlaps
        min_index = new_bases[oo - 1]
        max_index = new_bases[oo] - 1

        S1 = (numpy.array(bmus[min_index:max_index], int)).copy()
        avg_overlap.append(an_exp.get_overlap(S1))

        fm_val = []
        oc_val = []
        jac_val = []
        for rapetitions in xrange(exp_reps):

            S2 = S1.copy()
            S2 = an_exp.change_bmus(S2, f)

            # get rid of empty clusters and unassigned patterns
            S1_new, S2_new = an_exp.clean_up_bmuses(S1, S2)

            avg_overlap_S2.append(an_exp.get_overlap(S2))

            M = S1_new.dot(S2_new.transpose())

            # calculate indexes
            oc_temp = oc(M, S1_new, S2_new)
            fm_temp = fm(M, S1_new.shape[1])
            jac_temp = jac(M, S1_new.shape[1])

            # repetitions of a given fixed set of parameters
            # this defines a boxplot of the index
            fm_val.append(fm_temp)
            oc_val.append(oc_temp)
            jac_val.append(jac_temp)
            print "\toverlap: %d -> FM: %f \t OC: %f  \t JAC: %f" % (oo, fm_temp, oc_temp, jac_temp)

        fm_slice.append(fm_val)
        oc_slice.append(oc_val)
        jac_slice.append(jac_val)

    print "--------------\n\n"
    results_fm.append(fm_slice)
    results_oc.append(oc_slice)
    results_jac.append(jac_slice)

    avg_overlap_sorted, fm_slice_sorted = an_exp.sort_plot_values_by_overlap(avg_overlap, fm_slice)
    avg_overlap_sorted, oc_slice_sorted = an_exp.sort_plot_values_by_overlap(avg_overlap, oc_slice)

    # plotting results
    filename = "results_" + str(f) + "f.png"
    plot, ax = plt.subplots(figsize=(10,7))

    bp1_fm = plt.boxplot(fm_slice,  notch=False, patch_artist=False, showmeans=True, meanline=True)
    bp2_oc = plt.boxplot(oc_slice, notch=False, patch_artist=False)

    plt.setp(bp1_fm['boxes'], color='red')
    plt.setp(bp1_fm['whiskers'], color='red')
    plt.setp(bp1_fm['medians'], color='black')
    plt.setp(bp1_fm['fliers'], color='red', marker='+')

    plt.setp(bp2_oc['boxes'], color='blue')
    plt.setp(bp2_oc['whiskers'], color='blue')
    plt.setp(bp2_oc['medians'], color='black')
    plt.setp(bp2_oc['fliers'], color='blue', marker='+')

    plt.ylim(-0.1, 1.1)
    plt.title("Data modification: " + str(int(f * 100)) + "% - Repetitions: " + str(exp_reps), fontsize=14, fontweight='bold')

    plt.ylabel('Similarity', color="black")
    plt.xlabel('Subsets of communities with increasing overlapping', color="black")
    plt.xticks(range(1, len(avg_overlap_sorted)+1), range(1, len(avg_overlap_sorted)+1), fontsize=10, rotation=0)
    ax.set_xticks(range(1,len(avg_overlap_sorted) + 1))

    hB, = plt.plot([1,1],'b-')
    hR, = plt.plot([1,1],'r-')
    hB.set_visible(False)
    hR.set_visible(False)

    pp.savefig(dpi=900)
    plt.show()
    plt.savefig(results_path + filename, dpi=900)
    plt.clf()

pp.close()
#exit(0)
