"""
Read two dataframes: donor and recipient lanague
Search for phonetic matches
Calculate their semantic similarity
Output a list of candidate loanwords
"""
import heapq
import re
from functools import lru_cache

def phonetic_matches(df_ad, df_rc):
    """
    Finds phonetic matches between the given donor and recipient TSV files.

    The function processes the donor and recipient data frames,
    compares the phonetic patterns,
    and returns the matched data as a string in TSV format.

    :param donor_tsv_name: The file path of the donor TSV file.
    :type donor_tsv_name: str
    :param recipient_tsv_name: The file path of the recipient TSV file.
    :type recipient_tsv_name: str
    :return: A string containing the matched data in TSV format, with the following columns:
             ID, loanID, adrcID, df, form, predicted, meaning.
    :rtype: str
    """

    phonmatch = "ID\tloanID\tadrcID\tdf\tform\tpredicted\tmeaning"
    match_id, loan_id = 0, 0

    for i, rcrow in enumerate(df_rc):
        print(f"{i+1}/{len(df_rc)} iterations completed", end="\r")
        for adrow in df_ad:
            for ad in adrow[5]:
                if re.match(rcrow[4], ad):
                    line1 = [str(loan_id), str(match_id), str(rcrow[0]),
                             "recipient", rcrow[2], rcrow[4], rcrow[3]]
                    loan_id += 1
                    line2 = [str(loan_id), str(match_id), str(adrow[0]),
                             "donor", adrow[2], ad, adrow[4]]
                    loan_id += 1
                    phonmatch += "\n" + "\t".join(line1) + "\n" + "\t".join(line2)
                    match_id += 1
                    break

    return phonmatch

def semantic_matches(phmtsv, get_semsim):
    """
    Calculate semantic similarity between pairs of rows in phmtsv
    using the function
    get_semsim, and add columns with the calculated similarity
    and the closest semantic match to each row.

    :param phmtsv: A list of lists where each sublist represents a
                   row of data. The first
                   sublist should contain the header row, and each
                   subsequent sublist
                   should contain the data for one row.
                   The meanings have to be in column 6.
                   They have to already be splitted to a list

    :type phmtsv: list of lists

    :param get_semsim: A function that calculates the semantic similarity
                       between two strings.
    :type get_semsim: function

    :return: A tab-separated string representing
             the top 1000 rows in phmtsv with the added
             columns for semantic similarity and closest
             semantic match, sorted in
             descending order by semantic similarity and
             ascending order by loanID.
    :rtype: str
    """

    # Calculate semantic similarity and add columns to output rows
    results = [phmtsv.pop(0) + ["semsim", "closest_sem"]]  # header
    for i in range(0, len(phmtsv), 2):  # skip header
        print(f"{i+1}/{len(phmtsv[1:])-1} iterations completed", end="\r")

        # calculate semantic similarity
        semsim = get_semsim(phmtsv[i][6], phmtsv[i+1][6])

        results.append(phmtsv[i] + [str(round(semsim[0], 2)), semsim[1]])
        results.append(phmtsv[i+1] + [str(round(semsim[0], 2)), semsim[2]])

    # Sort results in descending order by semsim and ascending order by loanID
    sorted1000 = heapq.nlargest(
        1000, results[1:], key=lambda x: (float(x[7]), int(x[0]))
        )

    # Write results to output file
    lines = "\t".join(results[0])
    for row in sorted1000:
        lines += "\n" + "\t".join(row)
    return lines
