from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import e

import re
import os
import sys
import json
import tempfile
import subprocess
import operator
import collections
import conversion
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import shutil


# ENTITY_MENTION_REGEX = re.compile(r".*entity\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+).*", re.DOTALL)
# TR_MENTION_REGEX = re.compile(r".*transformed\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+).*", re.DOTALL)
# TA_MENTION_REGEX = re.compile(r".*reaction-associated\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+).*", re.DOTALL)
# WU_MENTION_REGEX = re.compile(r".*work-up\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+).*", re.DOTALL)
# CT_MENTION_REGEX = re.compile(r".*contained\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+).*", re.DOTALL)


def get_token_to_span_index_start(span_and_tokens, token_index):
    for st in span_and_tokens:
        if len(st) == 0:
            continue
        if st[3] == token_index:
            return st[0]
        
def get_token_to_span_index_end(span_and_tokens, token_index):
    for st in span_and_tokens:
        if len(st) == 0:
            continue
        if st[3] == token_index:
            return st[1]    
    

def training_format_to_brat(gold_path, exmaple, prediction_path, prediction):
## processing one file here 
    doc_key = exmaple["doc_key"]
    original_gold_example_brat_txt_path = os.path.join(gold_path, doc_key + ".txt")
    # token_path_per = token_path + doc_key + ".conll"
    
  #  print("original_gold_example_brat_txt_path:   ", original_gold_example_brat_txt_path)
  #  print("token_path_per:   ", token_path_per)
    
    # span_and_tokens = conversion.align_span_and_token_from_tokenized_conll(original_gold_example_brat_txt_path, token_path_per)

    span_and_tokens = conversion.align_span_and_token(original_gold_example_brat_txt_path)

    original_gold_example_brat_txt = None
    with open(original_gold_example_brat_txt_path, "r") as fr:
      original_gold_example_brat_txt = fr.read()

# span_and_tokens
 # =[[0, 7, 'Example', 0],
 #[8, 10, '50', 1],
 #[10, 11, '.', 2]
 #[] ]
    
#   now need to convert token_index to span_index
    prediction_token_anaphor_starts = prediction[0]
    prediction_token_anaphor_ends = prediction[1]
    prediction_token_antecedent_starts = prediction[2]
    prediction_token_antecedent_ends = prediction[3]
    prediction_labels = prediction[4]
    
    # prediction_token_path = prediction_path + doc_key + ".token"
    # with open(prediction_token_path, "w") as fw: 
    #   for index in range(len(prediction_token_anaphor_starts)):
    #     fw.write("(")
    #     fw.write(str(prediction_token_anaphor_starts[index]))
    #     fw.write(",")
    #     fw.write(str(prediction_token_anaphor_ends[index]))
    #     fw.write(")\t(")
    #     fw.write(str(prediction_token_antecedent_starts[index]))
    #     fw.write(",")
    #     fw.write(str(prediction_token_antecedent_ends[index]))
    #     fw.write(")\n")
    
    prediction_span_anaphor_starts = [get_token_to_span_index_start(span_and_tokens, token_index) for token_index in prediction_token_anaphor_starts]
    prediction_span_anaphor_ends = [get_token_to_span_index_end(span_and_tokens, token_index) for token_index in prediction_token_anaphor_ends]  # note that it is [] so no need look back to last token
    prediction_span_antecedent_starts = [get_token_to_span_index_start(span_and_tokens, token_index) for token_index in prediction_token_antecedent_starts]
    prediction_span_antecedent_ends = [get_token_to_span_index_end(span_and_tokens, token_index) for token_index in prediction_token_antecedent_ends]  # note that it is [] so no need look back to last token
    
    # so now we have the span!
    
    # start brat construction! 
    # mention: 
    # Tx (\t) entity_type (space) span_start (space) span_end (\t) text_content
    # T8	entity 528 545	The organic phase
    
    # in our case, antecedent is "entity" and anaphor has its bridging label lower-form. 
    
    # relation:
    # Rx (\t) relation_type (space) Arg1:Tx_anaphor (spance) Arg2:Tx_antecedent (\t\n)
    # R12	Coreference Arg1:T13 Arg2:T12	    
    
    #labels_types = np.unique(np.array(prediction_labels))
    
    #type_anaphor = {lab: lab.lower() for lab in labels_types}
    
    # pair start and end for anaphor and antecedent
    prediction_span_anaphor = [(s,e) for s,e in zip(prediction_span_anaphor_starts, prediction_span_anaphor_ends)]
    prediction_span_antecedent = [(s,e) for s,e in zip(prediction_span_antecedent_starts, prediction_span_antecedent_ends)]
    
    #should not but I'm doing it lol
    def remove_redundancy_span_pair(prediction_span_anaphor, prediction_span_antecedent, prediction_labels):
        prediction_pair = [(s,e) for s,e in zip(prediction_span_anaphor, prediction_span_antecedent)]
        prediction_span_anaphor_temp = []
        prediction_span_antecedent_temp = []
        prediction_labels_temp = []
        prediction_pair_temp = []
        for index, span_pair in enumerate(prediction_pair):
            if span_pair not in prediction_pair_temp:
                prediction_pair_temp.append(span_pair)
                
                prediction_span_anaphor_temp.append(prediction_span_anaphor[index])
                prediction_span_antecedent_temp.append(prediction_span_antecedent[index])
                prediction_labels_temp.append(prediction_labels[index])
        
        return prediction_span_anaphor_temp, prediction_span_antecedent_temp, prediction_labels_temp
        
    # recalling this fuction - should not but I'm doing it lol
    # prediction_span_anaphor, prediction_span_antecedent, prediction_labels = remove_redundancy_span_pair(prediction_span_anaphor, prediction_span_antecedent, prediction_labels)
    
    # prediction_span_path = prediction_path + doc_key + ".span"
    # with open(prediction_span_path, "w") as fw: 
    #   for index in range(len(prediction_span_anaphor)):
    #     fw.write(str(prediction_span_anaphor[index]))
    #     fw.write("\t")
    #     fw.write(str(prediction_span_antecedent[index]))
    #     fw.write("\n")
    
    
    #sort out antecedent first and remove redundancy
    prediction_span_antecedent_mapping = {}
    prediction_span_index = 1   # Tx start from 1 not 0
    
    for span_antecedent in prediction_span_antecedent:
      if span_antecedent not in prediction_span_antecedent_mapping.keys():
          prediction_span_antecedent_mapping[span_antecedent] = prediction_span_index
          prediction_span_index += 1
    
    #sort out anaphor first and remove redundancy
    prediction_span_anaphor_mapping = {} # it is related to the label so it need label_layer
    for label in np.unique(np.array(prediction_labels)):
        prediction_span_anaphor_mapping[label]={}  
    
    for index, span_anaphor in enumerate(prediction_span_anaphor):
      if span_anaphor not in prediction_span_anaphor_mapping[prediction_labels[index]].keys():
          prediction_span_anaphor_mapping[prediction_labels[index]][span_anaphor] = prediction_span_index
          prediction_span_index += 1
          
    prediction_ann = os.path.join(prediction_path, doc_key + ".ann")
    with open(prediction_ann, "w") as fw:
      # write the antecedent first
      for span_antecedent in prediction_span_antecedent_mapping.keys():
        antecedent_template = "T%s\t%s %s %s\t%s\n"%(str(prediction_span_antecedent_mapping[span_antecedent]), "entity", str(span_antecedent[0]), str(span_antecedent[1]), original_gold_example_brat_txt[span_antecedent[0]:span_antecedent[1]].replace('\n', ' ')) # need to replace \n 
        fw.write(antecedent_template)  
        
      # fit the anaphor
      for label in prediction_span_anaphor_mapping.keys():
          for span_anaphor in prediction_span_anaphor_mapping[label].keys():
              anaphor_template = "T%s\t%s %s %s\t%s\n"%(str(prediction_span_anaphor_mapping[label][span_anaphor]), label.lower(), str(span_anaphor[0]), str(span_anaphor[1]), original_gold_example_brat_txt[span_anaphor[0]:span_anaphor[1]].replace('\n', ' ')) ## need to lower() the label  # need to replace \n
              fw.write(anaphor_template)
      
      #  and relation
      prediction_relation_index = 1  # Rx start from 1 not 0
      for index, label in enumerate(prediction_labels):
        relation_template ="R%s\t%s Arg1:T%s Arg2:T%s\t\n"%(str(prediction_relation_index), label, str(prediction_span_anaphor_mapping[label][prediction_span_anaphor[index]]), str(prediction_span_antecedent_mapping[prediction_span_antecedent[index]]))
        fw.write(relation_template)
        prediction_relation_index+=1
    
    # also need to write txt
    prediction_txt = os.path.join(prediction_path, doc_key + ".txt")
    with open(prediction_txt, "w") as fw:
        fw.write(original_gold_example_brat_txt)
    
 #   print("convertion from training to brat done!!!")    

def get_p_r_f1(tp, fp, fn):
    if tp+fp == 0:
        p = 0
    else:
        p = tp/(tp+fp)
    if tp+fn == 0:
        r = 0
    else:
        r = tp/(tp+fn)
    if p+r == 0:
        f1 = 0
    else:
        f1 = 2*(p*r)/(p+r)
    return (p, r, f1)  

def get_mention_matching_result(label, MENTION_RESULTS_REGEX, stdout, total_tp, total_fp, total_fn, mention_precision, mention_recall, mention_f1):
    mention_results_match = re.match(MENTION_RESULTS_REGEX, stdout)
    
    total_tp += float(mention_results_match.group(1))
    total_fp += float(mention_results_match.group(2))
    total_fn += float(mention_results_match.group(3))
    
    mention_precision[label] = float(mention_results_match.group(4))
    mention_recall[label] = float(mention_results_match.group(5))
    mention_f1[label] = float(mention_results_match.group(6))
    
    return total_tp, total_fp, total_fn, mention_precision, mention_recall, mention_f1
    
def official_brat_eval_mention(eval_tool_path, gold_path, prediction_path, exact_matching, label, official_stdout=False):
    cmd = ["java -cp "+ eval_tool_path +"brateval/target/BRATEval-0.1.0-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareEntities "+ prediction_path +" "+ gold_path +" "+ exact_matching]  
    
#     print("cmd: ", cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    #process = subprocess.run(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    
    stdout = stdout.decode("utf-8")
#     print("stdout: ", stdout)
#     print("stderr: ", stderr)

    if stderr is not None:
      print(stderr)

    if official_stdout:
      print("Official result for {}".format(metric))
      print(stdout)

    
#     we make use of stdout!!! 

    MENTION_RESULTS_REGEX = re.compile(r".*%s\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+)\|([0-9.]+).*"%(label), re.DOTALL)
    
    mention_results_match = re.match(MENTION_RESULTS_REGEX, stdout)
    
    if mention_results_match == None:
        print("get None type error in mention_results_match with label : ", label)
        print("return lable: ", label, "with (0, 0, 0)")
        return 0, 0, 0, (0, 0, 0)
    
    mention_tp = float(mention_results_match.group(1))
    mention_fp = float(mention_results_match.group(2))
    mention_fn = float(mention_results_match.group(3))
    
    mention_precision = float(mention_results_match.group(4))
    mention_recall = float(mention_results_match.group(5))
    mention_f1 = float(mention_results_match.group(6))
    
    return mention_tp, mention_fp, mention_fn, (mention_precision, mention_recall, mention_f1)

def official_brat_eval_relation(eval_tool_path, gold_path, prediction_path, exact_matching, label, hop_evaluation=False, verbose="false", official_stdout=False):
    cmd = None
    if hop_evaluation:
      cmd = ["java -cp "+ eval_tool_path +"brateval/target/BRATEval-0.1.0-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareRelationsHop "+ prediction_path +" "+ gold_path +" "+ exact_matching+ " "+ verbose]
    else:
      cmd = ["java -cp "+ eval_tool_path +"brateval/target/BRATEval-0.1.0-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareRelations "+ prediction_path +" "+ gold_path +" "+ exact_matching+ " "+ verbose] 
    
#     print("cmd: ", cmd)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    #process = subprocess.run(cmd, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    process.wait()
    
    stdout = stdout.decode("utf-8")
#     print("stdout: ", stdout)
#     print("stderr: ", stderr)

    if stderr is not None:
      print(stderr)

    if official_stdout:
      print("Official result for {}".format(metric))
      print(stdout)
        
#     we make use of stdout!!! 

    RELATION_RESULTS_REGEX = re.compile(r".*%s\|.+?\|.+?\|tp:([0-9.]+)\|fp:([0-9.]+)\|fn:([0-9.]+)\|precision:([0-9.]+)\|recall:([0-9.]+)\|f1:([0-9.]+).*"%(label), re.DOTALL)
    
    relation_results_match = re.match(RELATION_RESULTS_REGEX, stdout)
    if relation_results_match == None:
        print("get None type error in relation_results_match with label: ", label)
        print("return lable: ", label, "with (0, 0, 0)")
        return 0, 0, 0, (0, 0, 0)
              
    relation_tp = float(relation_results_match.group(1))
    relation_fp = float(relation_results_match.group(2))
    relation_fn = float(relation_results_match.group(3))
    
    relation_precision = float(relation_results_match.group(4))
    relation_recall = float(relation_results_match.group(5))
    relation_f1 = float(relation_results_match.group(6))

    return relation_tp, relation_fp, relation_fn, (relation_precision, relation_recall, relation_f1)


def evaluate_brat(eval_tool_path, gold_path, predictions_bridging, predictions_coref, training_setting, predictions_original_jsonlines, official_stdout=False):

    name = sys.argv[1]
    #  now need to create the correct gold set for evaluation 
    eval_ids = [example["doc_key"] for example in predictions_original_jsonlines]

    store_gold_path = os.path.join("./gold/", name)

    # clear the eval file 
    if os.path.exists(store_gold_path): 
        shutil.rmtree(store_gold_path, ignore_errors=True)
    os.makedirs(store_gold_path)

    # now move the used gold file to the store file direction
    for e_ids in eval_ids:
        with open(os.path.join(gold_path, e_ids+".ann"), "r") as fr:
            with open(os.path.join(store_gold_path, e_ids+".ann"), "w") as fw:
                fw.write(fr.read())
        with open(os.path.join(gold_path, e_ids+".txt"), "r") as fr:
            with open(os.path.join(store_gold_path, e_ids+".txt"), "w") as fw:
                fw.write(fr.read())

    # re-assign the  gold path for evaluation
    gold_path = store_gold_path

    
    prediction_path = os.path.join("./prediction/", name)
    if not os.path.exists(prediction_path): 
        os.makedirs(prediction_path)

    # with tempfile.NamedTemporaryFile(delete=False, mode="w") as prediction_file:
    # with tempfile.TemporaryDirectory() as prediction_path:
    if True:
        for example in predictions_original_jsonlines:
        
            if training_setting == "joint_training":
                predictions = predictions_bridging[example["doc_key"]]
                prediction_coref = predictions_coref[example["doc_key"]]
                for i in range(len(predictions)):
                    predictions[i].extend(prediction_coref[i])
                    
            elif training_setting == "bridging":
                predictions = predictions_bridging[example["doc_key"]]
                
            elif training_setting == "coreference":
                predictions = predictions_coref[example["doc_key"]]

            training_format_to_brat(gold_path, example, prediction_path, predictions)
    #   print("done with transforming")
        
        exact_matching = "true"
        #exact_matching = "false"
        print("\nexact_matching: ", exact_matching)
        labels = None
        
        mention_results = {}
        relation_results = {}
        mention_tp = 0
        mention_fp = 0
        mention_fn = 0
        relation_tp = 0
        relation_fp = 0
        relation_fn = 0
        
       # bridging or joint training evaluation
        if training_setting == "bridging" or training_setting == "joint_training":
            # labels = ["transformed", "reaction-associated", "work-up", "contained"] # do not consider "entity" label
            labels = ["transformed", "ingredient_without-state-change_associated", "ingredient_with-state-change_associated"]
            for label in labels:
            #  print("label: ", label)
                tp,fp,fn, mention_results[label] = official_brat_eval_mention(eval_tool_path, gold_path, prediction_path, exact_matching, label, official_stdout=False)
                mention_tp += tp
                mention_fp += fp
                mention_fn += fn
                
                # defalut hop is false
                tp,fp,fn, relation_results[label.capitalize()] = official_brat_eval_relation(eval_tool_path, gold_path, prediction_path, exact_matching, label.capitalize(), hop_evaluation=False, verbose="false", official_stdout=False)
                relation_tp += tp
                relation_fp += fp
                relation_fn += fn
                
            mention_results['all_bridging'] = get_p_r_f1(mention_tp, mention_fp, mention_fn)
            relation_results['All_bridging'] = get_p_r_f1(relation_tp, relation_fp, relation_fn)
        
        #print("\nbridging evaluation done")
        
        # coref or joint training evaluation
        if training_setting == "coreference" or training_setting == "joint_training":
            tp,fp,fn, mention_results["coreference"] = official_brat_eval_mention(eval_tool_path, gold_path, prediction_path, exact_matching, "coreference", official_stdout=False)
            mention_tp += tp
            mention_fp += fp
            mention_fn += fn
            tp,fp,fn, relation_results["Coreference"] = official_brat_eval_relation(eval_tool_path, gold_path, prediction_path, exact_matching, "Coreference", hop_evaluation=False, verbose="false", official_stdout=False)
            relation_tp += tp
            relation_fp += fp
            relation_fn += fn
            
            _, _, _, mention_results["coreference_hop"] = official_brat_eval_mention(eval_tool_path, gold_path, prediction_path, exact_matching, "coreference", official_stdout=False)
            _, _, _, relation_results["Coreference_hop"] = official_brat_eval_relation(eval_tool_path, gold_path, prediction_path, exact_matching, "Coreference", hop_evaluation=True, verbose="false", official_stdout=False)
            
    #  print("\ncoref evaluation done")
        
        if training_setting == "joint_training":
            mention_results['all_joint'] = get_p_r_f1(mention_tp, mention_fp, mention_fn)
            relation_results['All_joint'] = get_p_r_f1(relation_tp, relation_fp, relation_fn)

    #  print("\njoint train evaluation done")
        
        if training_setting == "bridging":
            mention_results['all'] = mention_results['all_bridging']
            relation_results['All'] = relation_results['All_bridging']
            
        elif training_setting == "coreference":
            mention_results['all'] = mention_results['coreference']
            relation_results['All'] = relation_results['Coreference']
            
        elif training_setting == "joint_training":
            mention_results['all'] = mention_results['all_joint']
            relation_results['All'] = relation_results['All_joint']
            
        return mention_results, relation_results

        
if __name__ == "__main__":
    #gold_path = "/home/biaoyan/data/Chemical/chemical-patents/annotated_40_snippets/"
    gold_path = "/data/scratch/projects/punim0478/byron/Chemical/chemical-patents/test/"
    prediction_path = "/data/scratch/projects/punim0478/byron/Chemical/e2e-high-CRAFT-Chemical/prediction/"
    exact_matching = "true"
    label = "Transformed"
    official_brat_eval_relation(gold_path, prediction_path, exact_matching, label.capitalize(), hop_evaluation=False, verbose="false", official_stdout=False)
    label = "Coreference"
    official_brat_eval_relation(gold_path, prediction_path, exact_matching, label.capitalize(), hop_evaluation=True, verbose="false", official_stdout=False)
#     java -cp /home/biaoyan/data/Chemical/e2e-high-CRAFT-Chemical/brateval/target/BRATEval-0.1.0-SNAPSHOT.jar au.com.nicta.csp.brateval.CompareEntities /home/biaoyan/data/Chemical/e2e-high-CRAFT-Chemical/prediction/ /home/biaoyan/data/Chemical/chemical-patents/test true
