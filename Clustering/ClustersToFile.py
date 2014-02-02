from collections import defaultdict
import Settings
import Entropy

def compute_entropy(codes_per_cluster):
    d = dict()
    for label, codes in codes_per_cluster.items():
        d[label] = Entropy.entropy(codes)
    return d 

def aggregate_labels(codes_per_document, lst_labels):
    code_tally_by_cluster = defaultdict(lambda: defaultdict(int))
    codes_per_cluster = defaultdict(list)
    
    for i, labels in enumerate(lst_labels):
        for label in labels:
            for code in codes_per_document[i]:
                code_tally_by_cluster[label][code] = code_tally_by_cluster[label][code] + 1
                codes_per_cluster[label].append(code)
        
    return (code_tally_by_cluster, codes_per_cluster)

def clusters_to_file(file_name, cluster_labels, codes_per_document, results_sub_folder):
    """ 
        Takes a list of cluster labels, and a list of sets of codes for that document
        and dumps the codes per cluster to a file, one cluster per row, sorted by freq (desc)
        results_sub_folder = results sub folder
    """

    if len(cluster_labels) == 0:
        raise Exception("No lst_labels passed into clusters_to_file")
    
    # Handle both lists of atomic lst_labels, and lists of iterables
    # This becomes a list of list of labels
    if not '__iter__' in dir(cluster_labels[0]):
        lst_labels = [[item] for item in cluster_labels]
    else:
        lst_labels = cluster_labels[:]
    
    settings = Settings.Settings()
    results_dir = settings.results_directory + results_sub_folder + "\\"

    # For every document, get all cluster labels
    # and all codes, and tally each code - label combo
    code_tally_by_cluster, codes_per_cluster = aggregate_labels(codes_per_document, lst_labels)
    entropy_per_cluster = compute_entropy(codes_per_cluster)
    
    # Open
    handle = open(results_dir + file_name, "w+")
    
    # Write Contents
    write_header(handle)
    
    most_freq_codes = write_body(handle, code_tally_by_cluster, codes_per_cluster, entropy_per_cluster)
    write_footer(handle, most_freq_codes, entropy_per_cluster)

    # Close
    handle.close()

def write_header(handle):
    handle.write("Cluster Label,Unique Code Count,Total Codes,Entropy\n")

def write_body(handle, code_tally_by_cluster, codes_per_cluster, entropy_per_cluster):
    most_freq_codes = defaultdict(set)
    for label, code_freq in code_tally_by_cluster.items():
        # Sort by freq desc
        freq = sorted(code_freq.items(), key=lambda kvp:kvp[1], reverse=True)
        # CLuster Label
        handle.write(str(label))
        # Unique Code count
        handle.write("," + str(len(freq)))        
        # Total Codes
        code_count = len(codes_per_cluster[label])
        handle.write("," + (str(code_count)))        
        # Entropy
        handle.write("," + str(entropy_per_cluster[label]))

        # Codes with counts
        for i, (code, count) in enumerate(freq):
            handle.write("," + code + "," + str(count))
            most_freq_codes[i].add(code)
        
        handle.write("\n")
    return most_freq_codes

def write_footer(handle, most_freq_codes, entropy_per_cluster):
    handle.write("\nCode count per column:")
    #Skip the first three cols
    handle.write(",,,,") 
    # Get a cumulative count per column by taking the union over the set
    # of all codes for the col
    union = set()
    union_count = []
    for i, codes in sorted(most_freq_codes.items(), key=lambda item:item[0]):
        handle.write(str(len(codes)))
        if i < (len(most_freq_codes) - 1):
            handle.write(",,")
        union = union.union(codes)
        union_count.append(len(union))
    
    handle.write("\nCumulative Code count per column:")
    handle.write(",,,,") 
    for i, count in enumerate(union_count):
        handle.write(str(count))
        if i < (len(union_count) - 1):
            handle.write(",,")
    
    total_entropy = sum(entropy_per_cluster.values())
    mean_entropy = total_entropy / len(entropy_per_cluster.values())

    handle.write("\nTotal Entropy:,,,{0}\nMean Entropy per cluster:,,,{1}"
          .format(str(total_entropy).rjust(10), str(mean_entropy).rjust(10)))
    
