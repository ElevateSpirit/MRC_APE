tags2id = {'O': 0, 'B-Review': 1, 'I-Review': 2, 'E-Review': 3, 'S-Review': 4,
           'B-Reply': 1, 'I-Reply': 2, 'E-Reply': 3, 'S-Reply': 4,
           'B': 1, 'I': 2, 'E': 3, 'S': 4}

def spans_to_tags(spans, seq_len):
    tags = [tags2id['O']] * seq_len
    for span in spans:
        tags[span[0]] = tags2id['B']
        tags[span[0]:span[1]+1] = [tags2id['I']] * (span[1]-span[0]+1)
        if span[0] == span[1]:
            tags[span[0]] = tags2id['S']
        else:
            tags[span[0]] = tags2id['B']
            tags[span[1]] = tags2id['E']
    return tags


def get_arg_span(bioes_tags):
    start, end = None, None
    arguments = []
    in_entity_flag = False
    for idx, tag in enumerate(bioes_tags):
        if in_entity_flag == False:
            if tag == 1: # B
                in_entity_flag = True
                start = idx
            elif tag == 4: # S
                start = idx
                end = idx
                arguments.append((start, end))
                start = None
                end = None
        else:
            if tag == 0: # O
                in_entity_flag = False
                start = None
                end = None
            elif tag == 1: # B
                in_entity_flag = True
                start = idx
            elif tag == 3: # E
                in_entity_flag = False
                end = idx
                arguments.append((start, end))
                start = None
                end = None
            elif tag == 4: # S
                in_entity_flag = False
                start = idx
                end = idx
                arguments.append((start, end))
                start = None
                end = None
    return arguments



def extract_span_arguments_yi(match_labels,start_labels,end_labels):
    arguments_list = []
    for match_l, start_l, end_l in zip(match_labels,start_labels,end_labels):
        arguments = extract_spans_yi( start_l, end_l,match_l)
        arguments_list.append(arguments)
    return arguments_list

def extract_spans_yi(start_pred, end_pred, match_pred):
    pseudo_input = "a"

    label_mask=[1]*len(start_pred)
    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B"
    for end_item in end_positions:
        bmes_labels[end_item] = f"E"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"I"
            else:
                bmes_labels[tmp_end] = f"S"

    tags = get_arg_span([tags2id[label] for label in bmes_labels])

    return tags