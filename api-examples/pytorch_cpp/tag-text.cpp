#include <torch/script.h>
#include "json.hpp"
using json = nlohmann::json;

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <glob.h>
#include <stdlib.h>

const long UNK = 3;
const long PAD = 0;


void lower_case(const std::string& word, std::string& dst) {
    dst.resize(word.size());
    std::transform(
        word.begin(),
        word.end(),
        dst.begin(),
        ::tolower
    );
}


void find_vocab_file(const std::string& bundle, const std::string& vocab, std::string& dst) {
    glob_t glob_result;
    std::stringstream file_name;
    file_name << bundle << "/vocabs-" << vocab << "-*.json";
    glob(file_name.str().c_str(), GLOB_TILDE, NULL, &glob_result);
    dst = glob_result.gl_pathv[0];
}


void find_label_file(const std::string& bundle, std::string& dst) {
    glob_t glob_result;
    std::stringstream file_name;
    file_name << bundle << "*.labels";
    glob(file_name.str().c_str(), GLOB_TILDE, NULL, &glob_result);
    dst = glob_result.gl_pathv[0];
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tag-text export_bundle tokens...\n";
        exit(1);
    }

    std::string bundle(argv[1]);
    std::vector<std::string> args(argv + 2, argv + argc);

    std::stringstream model_file;
    model_file << bundle << "/model.pt";
    std::shared_ptr<torch::jit::script::Module> model = torch::jit::load(model_file.str());
    assert(model != nullptr);

    std::cout << "Reading model metadata." << std::endl;
    std::stringstream metadata_file;
    metadata_file << bundle << "/model.assets";
    std::ifstream i(metadata_file.str());
    json metadata;
    i >> metadata;
    std::vector<std::string> order = metadata["inputs"];

    std::cout << "Reading vocabularies." << std::endl;
    std::unordered_map<std::string, json> vocabs;
    for (int i = 0, sz = order.size(); i < sz; i++) {
        std::string file_name;
        find_vocab_file(bundle, order[i], file_name);
        std::ifstream file(file_name);
        json vocab;
        file >> vocab;
        vocabs.insert({order[i], vocab});
    }

    std::cout << "Reading label vocab." << std::endl;
    std::string file_name;
    find_label_file(bundle, file_name);
    std::ifstream label_file(file_name);
    json label_to_idx;
    label_file >> label_to_idx;
    // Create a revlut
    std::unordered_map<int, std::string> label_vocab;
    for (json::iterator it = label_to_idx.begin(), end = label_to_idx.end(); it != end; ++it) {
        label_vocab.insert(std::pair<int, std::string>(it.value(), it.key()));
    }

    std::vector<torch::jit::IValue> input;
    std::vector<torch::jit::IValue> features;

    std::cout << "Building Features.\n";
    int mxwlen = 0;
    int args_sz = args.size();
    for (int i = 0, sz = order.size(); i < sz; i++) {
        std::vector<long> feature;
        std::string feat = order[i];
        // Create character based features
        if (feat == "char") {
            for (int j = 0; j < args_sz; j++) {
                int curr_sz = args[j].size();
                if (mxwlen < curr_sz) {
                    mxwlen = curr_sz;
                }
            }
            for (int j = 0; j < args_sz; j++) {
                // Build it as a single long vector and reshape the tensor
                int curr_sz = args[j].size();
                for (int k = 0; k < mxwlen; k++) {
                    if (k < curr_sz) {
                        std::string c(1, args[j][k]);
                        auto idx = vocabs[order[i]].find(c);
                        if (idx != vocabs[order[i]].end()) {
                            feature.push_back(vocabs[order[i]][c]);
                        } else {
                            feature.push_back(UNK);
                        }
                    } else {
                        feature.push_back(PAD);
                    }
                }
            }
            torch::Tensor f = torch::tensor(feature, torch::dtype(torch::kInt64));
            f = f.reshape({1, (int)args.size(), mxwlen});
            features.push_back(f);
        } else {
            // build out the token1d features
            for (int j = 0; j < args_sz; j++) {
                std::string lc;
                lower_case(args[j], lc);
                auto idx = vocabs[order[i]].find(lc);
                if (idx != vocabs[order[i]].end()) {
                    feature.push_back((long)vocabs[order[i]][lc]);
                } else {
                    feature.push_back(UNK);
                }
            }
            auto f = torch::tensor(feature, torch::dtype(torch::kInt64));
            f = f.unsqueeze(0);
            features.push_back(f);
        }
    }
    input.push_back(torch::jit::Tuple::create(features));

    auto lengths = torch::full({1}, (int)args_sz, torch::dtype(torch::kInt64));
    input.push_back(lengths);

    std::cout << "Tagging." << std::endl;
    auto output = model->forward(input).toTensor();

    std::vector<long> outputs(output.data<long>(), output.data<long>() + output.numel());

    std::cout << "Outputs:\n";
    for (int i = 0; i < args_sz; i++) {
        std::string space(mxwlen - args[i].size() + 2, ' ');
        std::cout << args[i] << space << label_vocab[outputs[i]] << "\n";
    }
}
