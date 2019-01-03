import pandas as pd
import json
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

# Load the original dataset
node_details_clean = pd.read_json(
    "/Users/avgupta/Downloads/node-package-details-clean.json", lines=True)

# generate clean data, fill Nan
print("generate clean data, fill Nan")
node_details_trim = node_details_clean[["dependencies", "name", "keywords"]]
node_details_trim[['dependencies', 'keywords']]   = node_details_trim[[
    'dependencies', 'keywords']].fillna(-1)
print("Delete node details clean")
del(node_details_clean)

print("Save clean data")
# save the clean data
node_details_trim.to_csv("node_trim_clean.csv")

print("Generate the Package_Dependecies Dict")
# Generate the Package_Dependecies Dict
package_dependencies_dict = dict()
for _, row in node_details_trim.iterrows():
    key_package_name = row["name"].lower()
    values = row["dependencies"]
    if type(values) == int and values == -1:
        values = []
    values = [dep_name.lower() for dep_name in values]
    if key_package_name not in package_dependencies_dict:
        package_dependencies_dict[key_package_name] = list(set(values))
    else:
        current_list = package_dependencies_dict[key_package_name]
        current_list.extend(values)
        package_dependencies_dict[key_package_name] = list(set(current_list))

print("Generate the Package_Keywords Dict")
# Generate the Package_Keywords Dict
package_keywords_dict = dict()
unique_keywords_collection = set()
for _, row in node_details_trim.iterrows():
    key_package_name = row["name"].lower()
    values_keywords = row["keywords"]
    if type(values_keywords) == int and values_keywords == -1:
        values_keywords = []
    values = []
    for name in values_keywords:
        values.append(name.lower())
        unique_keywords_collection.add(name.lower())
    if key_package_name not in package_keywords_dict:
        package_keywords_dict[key_package_name] = list(set(values))
    else:
        current_list = package_keywords_dict[key_package_name]
        current_list.extend(values)
        package_keywords_dict[key_package_name] = list(set(current_list))


assert len(package_keywords_dict) == len(package_dependencies_dict)
print("Delete node_details trim")
del(node_details_trim)

print("Generate Unique_Keyword_Index_Map and Unique_Package_Name_Index_Map")
# Generate Unique_Keyword_Index_Map and Unique_Package_Name_Index_Map
unique_keywords_collection = list(unique_keywords_collection)
unique_packages_collection = package_dependencies_dict.keys()
map_keywords_dict = dict(
    zip(unique_keywords_collection, range(len(unique_keywords_collection))))
map_packages_dict = dict(
    zip(unique_packages_collection, range(len(unique_packages_collection))))
print("Delete unique key/pck collection")
del(unique_keywords_collection)
del(unique_packages_collection)

print("Save the Index_Map for future use")
# Save the Index_Map for future use
import json
with open("unique_package_index.json", "w") as f:
    json.dump(map_packages_dict, f)
with open("unique_keywords_index.json", "w") as f:
    json.dump(map_keywords_dict, f)

print("Genearate the Package_Aggregated_Tags_Dict")
# Genearate the Package_Aggregated_Tags_Dict
rating_matrix_dict = dict()
for package_name, current_tags in package_keywords_dict.items():
    for each_dep in package_dependencies_dict[package_name]:
        tags = package_keywords_dict.get(each_dep, [])
        current_tags.extend(tags)
    current_tags_index_wise = [map_keywords_dict[keyword]
                               for keyword in current_tags]
    rating_matrix_dict[map_packages_dict[package_name]] = list(
        set(current_tags_index_wise))

print("Delete apckage/dep/key dict")
del(package_dependencies_dict)
del(package_keywords_dict)

print("Generate Sparse Matrix using the Package_Aggregated_Tags_Dict")
# Generate Sparse Matrix using the Package_Aggregated_Tags_Dict

sparse_mat = sp.dok_matrix(
    (len(map_packages_dict), len(map_keywords_dict)), dtype=np.int64)
for package_id, tag_ids in rating_matrix_dict.items():
    sparse_mat[package_id, tag_ids] = 1

print("Generate Sparse Coordinate matrix")
# Generate the Sparse Cooridnate Matrix
sparse_coo = sparse_mat.tocoo()
# print("Delete sparse mat temp")
# del(sparse_mat)
indices = np.mat([sparse_coo.row, sparse_coo.col]).transpose()

print("Genearte the Sparse Tensor using Sparse Coo. Matrix")
# Genearte the Sparse Tensor using Sparse Coo. Matrix
rating_matrix = tf.SparseTensor(indices, sparse_coo.data, sparse_coo.shape)
print("Size of content matrix = {}".format(rating_matrix.get_shape()))
