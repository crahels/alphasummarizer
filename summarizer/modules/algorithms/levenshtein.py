from copy import deepcopy

def levenshtein_distance(word_1, word_2):
    word_1_length = len(word_1)
    word_2_length = len(word_2)

    # create two work vectors of integer distances
    v0 = [0] * (word_2_length + 1)
    v1 = [0] * (word_2_length + 1)

    # initialize v0 (the previous row of distances)
    # this row is A[0][i]: edit distance for an empty s
    # the distance is just the number of characters to delete from t
    for i in range(word_2_length + 1):
        v0[i] = i

    for i in range(word_1_length):
        # calculate v1 (current row distances) from the previous row v0

        # first element of v1 is A[i+1][0]
        # edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1

        # use formula to fill in the rest of the row
        for j in range(word_2_length):
            # calculating costs for A[i+1][j+1]
            deletion_cost = v0[j + 1] + 1
            insertion_cost = v1[j] + 1
            if word_1[i] == word_2[j]:
                substitution_cost = v0[j]
            else:
                substitution_cost = v0[j] + 1

            v1[j + 1] = min(deletion_cost, insertion_cost, substitution_cost)

        # copy v1 (current row) to v0 (previous row) for next iteration
        # swap v0 with v1
        v0 = deepcopy(v1)

    # after the last swap, the results of v1 are now in v0
    max_length = max(word_1_length, word_2_length)
    if (max_length > 0):
        return float(v0[word_2_length])/max_length
    else:
        return float(v0[word_2_length])
