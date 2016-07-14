"""
@author: dncampo
"""

import sys
import numpy
import random

class Youtube_experiments:
    """
    Class file that handles the YouTube dataset and prepares it for an
    overlap experiment
    """

    the_dataset = []
    comm_overlap = []
    sorted_overlaps = []
    bmus = []
    overlap_bmus = []

    def print_out_matrix(self, a_matrix):
        sys.stdout.write("[")
        for i in range(a_matrix.shape[0]):
            for j in range(a_matrix.shape[1]):
                sys.stdout.write("%d, " % (a_matrix[i, j]))
            sys.stdout.write(";\n")
        sys.stdout.write("]")


    def create_consecutives_ids(self, the_dataset):
        max_list = [max(x) for x in the_dataset]
        max_id = max(max_list)

        ids_presents = numpy.zeros(max_id + 1, bool)

        # vector containing the IDs that are present in the file
        for line in the_dataset:
            for a_id in line:
                ids_presents[a_id] = True

        # convert IDs
        old2newID = {}
        new2oldID = {}
        fpold2newid = open("old2new_id.txt", "w")
        fpnew2oldid = open("new2old_id.txt", "w")
        index = 1  # starts with ID=1
        for i in range(max_id + 1):
            if ids_presents[i]:
                old2newID[i] = index
                new2oldID[index] = i
                fpold2newid.write("%d\t%d\n" % (i, index))
                fpnew2oldid.write("%d\t%d\n" % (index, i))
                index += 1
        fpold2newid.close()
        fpnew2oldid.close()

        new_dataset = [[old2newID[y] for y in xx] for xx in the_dataset]
        self.the_dataset = new_dataset
        return

    # it returns the degree of overlap of each comm
    def get_overlaps(self):
        max_list = [max(x) for x in self.the_dataset]
        max_id = max(max_list)

        # number of times that each user appears
        id_count = numpy.zeros(max_id + 1, int)
        for line in self.the_dataset:
            for a_id in line:
                id_count[a_id] += 1

        comm_overlap = numpy.zeros(len(self.the_dataset), int)
        i = 0;
        for line in self.the_dataset:
            overlp = 0
            for a_id in line:
                if (id_count[a_id] > 1):
                    overlp += 1;
            comm_overlap[i] = overlp
            i += 1

        self.comm_overlap = comm_overlap
        return comm_overlap

    @property
    def get_sorted_overlaps(self):
        comm_sorted_overlap = {}

        i = 0
        for line in self.the_dataset:
            num_overlaps = self.comm_overlap[i]
            if not comm_sorted_overlap.has_key(num_overlaps):
                comm_sorted_overlap[num_overlaps] = []
            comm_sorted_overlap[num_overlaps].append(line)
            i += 1

        sorted(comm_sorted_overlap)

        self.sorted_overlaps = comm_sorted_overlap
        return comm_sorted_overlap


    # the_dataset is a dictionary
    def create_bmus(self):
        overlaps = self.sorted_overlaps.keys()
        overlaps.sort()
        plain_dataset = []
        overlap_plain_dataset = []

        for overl in overlaps:
            for comm in self.sorted_overlaps[overl]:
                plain_dataset.append(comm)
                overlap_plain_dataset.append(overl)


        # given that the IDs are already continuous, the max returns the number of users
        max_list = [max(x) for x in plain_dataset]
        users = max(max_list)

        comms = len(plain_dataset)

        bmus = numpy.zeros((comms, users), bool)
        for i in range(len(plain_dataset)):
            for j in range(len(plain_dataset[i])):
                user = plain_dataset[i][j]
                bmus[i, user - 1] = True;

        self.bmus = bmus
        self.overlap_bmus = overlap_plain_dataset
        return bmus, overlap_plain_dataset


    def change_bmus(self, bmus, f=0.1):
        changes = int(bmus.shape[1] * f)

        min_comm = 0
        min_user = 0
        max_comm = bmus.shape[0] - 1
        max_user = bmus.shape[1] - 1

        for i in range(changes):
            change_comm = random.randint(min_comm, max_comm)
            change_user = random.randint(min_user, max_user)

            # just add an f% (10% by default) of users to new communities
            bmus[change_comm, change_user] = True
        return bmus


    def are_zero_cols(self, bmus):
        return not ((bmus.sum(0) != 0).all())


    def are_zero_rows(self, bmus):
        return not ((bmus.sum(1) != 0).all())


    def remove_null_rows(self, an_array):
        """
        :param an_array: a collection, preferably an array
        :return: the same collection without its zero-rows
        """
        return numpy.delete(an_array, numpy.where(0 == an_array.sum(1)), 0)


    def remove_null_columns(self, an_array_1, an_array_2):
        """
        :param an_array_1: a collection, preferably an array
        :param an_array_2: a collection, preferably an array
        :return: both collections without the columns that are zero
        in any of them
        """
        # list of indexes to delete of each collection
        ar_1_zero_col = (numpy.where(0 == an_array_1.sum(0)))[0].tolist()
        ar_2_zero_col = (numpy.where(0 == an_array_2.sum(0)))[0].tolist()

        # merge the set of 'to-delete' indexes, without dupes
        to_delete = sorted(set(ar_1_zero_col + ar_2_zero_col))

        return numpy.delete(an_array_1, to_delete, 1), \
               numpy.delete(an_array_2, to_delete, 1)


    # delete rows and cols that are completely 0
    # i.e. empty communities or users that does not belong to a comm.
    def clean_up_bmuses(self, S1, S2):
        S1_zc, S2_zc = self.remove_null_columns(S1, S2)

        S1_zrzc = self.remove_null_rows(S1_zc)
        S2_zrzc = self.remove_null_rows(S2_zc)
        return S1_zrzc, S2_zrzc

    # split out the dataset
    def break_down_overlaps(self, considered_overlaps):
        zero = self.overlap_bmus.count(0) * 3
        breaks = []

        bases = [0]
        delta = 0
        min_index = 0

        actual = []

        for oo in considered_overlaps:

            delta = delta + self.overlap_bmus.count(oo)

            actual.append(oo)

            if (delta >= zero or oo == 0):
                breaks.append(actual)
                min_index = min_index + delta
                bases.append(min_index)
                delta = 0
                actual = []

        if (len(actual) > 0):
            min_index = min_index + delta + 1
            bases.append(min_index)
            breaks.append(actual)

        return bases, breaks

    def get_overlap(self, bmus):
        """
        :param bmus: bmus matrix
        :return: the relative avg. overlap for a given bmus
        To achieve this, the overlap of each comm is added and
        the, this accumulated value is divided by the number of
        considered comm.
        The overlap of each comm. is calculated as the number
        of its users that belong to other communities
        """

        row_sum = bmus.sum(0)
        for i in range(len(row_sum)):
            if row_sum[i] > 1:
                row_sum[i] = 1
            else:
                row_sum[i] = 0

        accum = numpy.zeros(bmus.shape[0])
        for i in range(bmus.shape[0]):
            accum[0] = bmus[0].dot(row_sum.transpose())

        return round(accum.sum() / bmus.shape[0], 2)

    def sort_plot_values_by_overlap(self, avg_overlap, index_slice):
        both_sorted_pair = zip(*sorted(zip(avg_overlap, index_slice)))
        avg_overlap_sorted = both_sorted_pair[0]
        index_sorted = both_sorted_pair[1]

        # converting tuple to array
        avg_overlap_sorted = numpy.array(avg_overlap_sorted)
        index_sorted = numpy.array(index_sorted)

        return avg_overlap_sorted.tolist(), index_sorted.tolist()


