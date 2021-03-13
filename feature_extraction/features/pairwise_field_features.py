#!/usr/bin/python3
# -*- coding: utf-8 -*-

import editdistance
import numpy as np
import pandas as pd 
from collections import OrderedDict
from scipy.stats import pearsonr, f_oneway, chi2_contingency, ks_2samp
from itertools import combinations
from time import time

from .type_detection import data_types, general_types
from .helpers import parse, get_unique, get_list_uniqueness, get_shared_elements, calculate_overlap


general_pairwise_features_list = [
    {'name': 'fid', 'type': 'id'},
    {'name': 'field_a_id', 'type': 'id'},
    {'name': 'field_b_id', 'type': 'id'},    
    {'name': 'pair_exists', 'type': 'boolean'},
    {'name': 'has_shared_elements', 'type': 'boolean'},
    {'name': 'num_shared_elements', 'type': 'numeric'},
    {'name': 'percent_shared_elements', 'type': 'numeric'}, 
    {'name': 'identical', 'type': 'boolean'},
    {'name': 'has_shared_unique_elements', 'type': 'boolean'},
    {'name': 'num_shared_unique_elements', 'type': 'numeric'},
    {'name': 'percent_shared_unique_elements', 'type': 'numeric'}, 
    {'name': 'identical_unique', 'type': 'boolean'},    
]

qq_pairwise_features_list = [
    {'name': 'correlation_value', 'type': 'numeric'},           # The Pearson correlation coefficient measures the linear relationship between two datasets
    {'name': 'correlation_p', 'type': 'numeric'},
    {'name': 'correlation_significant_005', 'type': 'boolean'},
    {'name': 'ks_statistic', 'type': 'numeric'},
    {'name': 'ks_p', 'type': 'numeric'},  
    {'name': 'ks_significant_005', 'type': 'boolean'},    
    {'name': 'percent_range_overlap', 'type': 'numeric'},
    {'name': 'has_range_overlap', 'type': 'numeric'},
]

cc_pairwise_features_list = [
    {'name': 'chi_sq_statistic', 'type': 'numeric'},        # 统计两个维度各种组合的组合数，进行卡方检验
    {'name': 'chi_sq_p', 'type': 'numeric'},                # 卡方检验用于判断样本分布与期望分布是否有显著差异，或者判断分类变量间是否相互关联或彼此独立，此处用列联表，属后者
    {'name': 'chi_sq_significant_005', 'type': 'numeric'},  # 若p值<0.05，拒绝原假说（两维度互相独立），证明两维度相互关联
    {'name': 'is_nested', 'type': 'boolean'}, 
    {'name': 'nestedness', 'type': 'numeric'},      # 两个维度的镶嵌程度，即一个为另一个的子分类的程度
    {'name': 'nestedness_95', 'type': 'boolean'}, 
]

cq_pairwise_features_list = [                                       # The one-way ANOVA tests the null hypothesis that
    {'name': 'one_way_anova_statistic', 'type': 'numeric'},         # two or more groups have the same population mean
    {'name': 'one_way_anova_p', 'type': 'numeric'},                 # 先把q维度的数值按c维度分组，测试每组均值是否相同（要求每组方差相同）
    {'name': 'one_way_anova_significant_005', 'type': 'boolean'},   # 若p值<0.05，拒绝原假说（各组均值相同），即至少有两组的均值不同
]

name_pairwise_features_list = [
    {'name': 'edit_distance', 'type': 'numeric'}, 
    {'name': 'normalized_edit_distance', 'type': 'numeric'},     
    {'name': 'has_shared_words', 'type': 'boolean'}, 
    {'name': 'num_shared_words', 'type': 'numeric'}, 
    {'name': 'percent_shared_words', 'type': 'numeric'},     
]

statistical_pairwise_features_list = \
    qq_pairwise_features_list + \
    cc_pairwise_features_list + \
    cq_pairwise_features_list

all_pairwise_features_list = \
    general_pairwise_features_list + \
    statistical_pairwise_features_list + \
    name_pairwise_features_list

def get_general_pairwise_features(a, b):
    r = OrderedDict([ (f['name'], None) for f in general_pairwise_features_list ])
    a_data = a['data']
    b_data = b['data']  
    a_unique_data = set(a['unique_data'])
    b_unique_data = set(b['unique_data'])      
    
    num_identical_elements = np.count_nonzero(a_data == b_data)
    r['has_shared_elements'] = (num_identical_elements > 0)
    r['num_shared_elements'] = num_identical_elements
    r['percent_shared_elements'] = num_identical_elements / len(a_data)
    r['identical'] = num_identical_elements == len(a_data)

    num_shared_unique_elements = len(a_unique_data.intersection(b_unique_data))
    r['has_shared_unique_elements'] = (num_shared_unique_elements > 0)
    r['num_shared_unique_elements'] = num_shared_unique_elements
    r['percent_shared_unique_elements'] = num_shared_unique_elements/ max(len(a_unique_data), len(b_unique_data))
    r['identical_unique'] = (a_unique_data == b_unique_data)  
    return r

def get_statistical_pairwise_features(a, b, MAX_GROUPS=50):
    r = OrderedDict([ (f['name'], None) for f in statistical_pairwise_features_list ])
    a_name = a['name']
    b_name = b['name']
    a_data = a['data']
    b_data = b['data']  

    # Match lengths
    min_len = min(len(a_data), len(b_data))
    a_data = a_data[:min_len]
    b_data = b_data[:min_len]       

    if (a['general_type'] == 'q' and b['general_type'] == 'q'):
        correlation_value, correlation_p = pearsonr(a_data, b_data)
        ks_statistic, ks_p = ks_2samp(a_data, b_data)
        has_overlap, overlap_percent = calculate_overlap(a_data, b_data)

        r['correlation_value'] = correlation_value
        r['correlation_p'] = correlation_p
        r['correlation_significant_005'] = (correlation_p < 0.05)

        r['ks_statistic'] = ks_statistic
        r['ks_p'] = ks_p
        r['ks_significant_005'] = (ks_p < 0.05)

        r['has_overlap'] = has_overlap
        r['overlap_percent'] = overlap_percent

    if (a['general_type'] == 'c' and b['general_type'] == 'c'):
        if len(get_unique(a_data)) > MAX_GROUPS or len(get_unique(a_data)) > MAX_GROUPS:
            return r
        df = pd.DataFrame({ a_name: a_data, b_name: b_data })
        ct = pd.crosstab(a_data, b_data)    # 交叉表，统计分组频率
        chi2_statistic, chi2_p, dof, exp_frequencies = chi2_contingency(ct)

        nestedness_values = []
        for parent, child in ([a, b], [b, a]):
            child_field_unique_corresponding_values = []
            unique_parent_field_values = parent['unique_data']
            for unique_parent_field_value in unique_parent_field_values:
                child_field_unique_corresponding_values.extend( set( df[ df[parent['name']] == unique_parent_field_value ][ child['name'] ] ) )
            nestedness = get_list_uniqueness(child_field_unique_corresponding_values)     
            nestedness_values.append(nestedness) 

        r['chi2_statistic'] = chi2_statistic
        r['chi2_p'] = chi2_p
        r['chi2_significant_005'] = (chi2_p < 0.05)
        r['nestedness'] = max(nestedness_values)
        r['nestedness_95'] == (nestedness > 0.95)

    if (a['general_type'] == 'q' and b['general_type'] == 'c') or (a['general_type'] == 'c' and b['general_type'] == 'q'):
        c_field = a
        q_field = b
        if a['general_type'] == 'q' and b['general_type'] == 'c':
            c_field = b
            q_field = a

        unique_c_field_values = get_unique(c_field['data'])
        if len(unique_c_field_values) <= MAX_GROUPS:
            df = pd.DataFrame({ a_name: a_data, b_name: b_data })
            group_values = [ df[df[c_field['name']] == v][q_field['name']] for v in unique_c_field_values ] # 按c维度分组
            anova_result = f_oneway(*group_values)  

            r['one_way_anova_statistic'] = anova_result.statistic
            r['one_way_anova_p'] = anova_result.pvalue
            r['one_way_anova_significant_005'] = (anova_result.pvalue < 0.05)
    
    return r

def get_name_pairwise_features(n1, n2, MAX_NAME_LENGTH=500):
    r = OrderedDict([ (f['name'], None) for f in name_pairwise_features_list ])

    if (len(n1) > MAX_NAME_LENGTH) or (len(n2) > MAX_NAME_LENGTH):
        return r
    edit_distance = editdistance.eval(n1, n2)
    normalized_edit_distance = edit_distance / max(len(n1), len(n2))
    n1_words = n1.split(' ')
    n2_words = n2.split(' ')
    shared_words = get_shared_elements(n1_words, n2_words)

    r['edit_distance'] = edit_distance
    r['normalized_edit_distance'] = normalized_edit_distance
    r['has_shared_words'] = (len(shared_words) > 0)
    r['num_shared_words'] = len(shared_words)
    r['percent_shared_words'] = len(shared_words) / max(len(n1), len(n2))
    return r


def extract_pairwise_field_features(field_data, single_field_features, fid, timeout=15, MAX_FIELDS=10):
    # Pre-process types from single field features
    num_fields = len(field_data)

    # Populate features dictionary. Structure:
    # [
    #     [2, 3, 4, 5]
    #     [3, 4, 5]
    #     [4, 5 ]
    #     [5]
    # ]
    all_pairwise_field_features = []
    for field_1_index in range(MAX_FIELDS - 1):
        field_1_pairwise_field_features = []
        for field_2_index in range(field_1_index + 1, MAX_FIELDS):
            pairwise_field_features = OrderedDict([ (f['name'], None) for f in all_pairwise_features_list ])
            field_1_pairwise_field_features.append(pairwise_field_features)
        all_pairwise_field_features.append(field_1_pairwise_field_features)

    for field_1_index in range(num_fields - 1):
        for field_2_index in range(0, num_fields - field_1_index - 1):
            absolute_field_2_index = field_1_index + field_2_index + 1
            a = field_data[field_1_index]
            b = field_data[absolute_field_2_index]

            all_pairwise_field_features[field_1_index][field_2_index]['pair_exists'] = True
            all_pairwise_field_features[field_1_index][field_2_index]['fid'] = fid
            all_pairwise_field_features[field_1_index][field_2_index]['field_a_id'] = field_1_index
            all_pairwise_field_features[field_1_index][field_2_index]['field_a_id'] = absolute_field_2_index

            general_pairwise_features = OrderedDict()
            name_pairwise_features = OrderedDict()
            statistical_pairwise_features = OrderedDict()

            try:
                start_time = time()
                while time() < (start_time + timeout):
                    name_pairwise_features = get_name_pairwise_features(a['name'], b['name'])
                    statistical_pairwise_features = get_statistical_pairwise_features(a, b)
                    general_pairwise_features = get_general_pairwise_features(a, b)
                    break
            except Exception as e:
                print('Error getting features for {} and {}'.format(a['name'], b['name']))
                pass

            for feature_set in [ general_pairwise_features, name_pairwise_features, statistical_pairwise_features]:
                for k, v in feature_set.items():
                    all_pairwise_field_features[field_1_index][field_2_index][k] = v
    return all_pairwise_field_features
