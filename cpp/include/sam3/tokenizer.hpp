// tokenizer.hpp
// Author: Jason Hughes
// Date:   2026
//
// CLIP BPE tokenizer for SAM3 text prompts.
// Identical interface to the clipper CLIPTokenizer.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>
#include <json/json.h>

namespace Sam3
{

class CLIPTokenizer
{
public:
    CLIPTokenizer() = default;
    /// @param merges_path  Path to merges.txt (BPE merge rules)
    /// @param vocab_path   Path to vocab.json  (token -> id mapping)
    CLIPTokenizer(const std::string& merges_path, const std::string& vocab_path);

    /// Tokenize a single text string into CLIP token IDs.
    /// Returns ids wrapped with SOT (49406) and EOT (49407) tokens.
    std::vector<int> tokenize(const std::string& text) const;

    // Accessors
    const std::vector<std::pair<std::string,std::string>>& getMerges() const;
    const std::unordered_map<std::string,int>&              getVocab()  const;
    int getPaddingToken() const { return 49407; }  // <|endoftext|>
    int getSOTToken()     const { return 49406; }  // <|startoftext|>

private:
    void loadMerges(const std::string& path);
    void loadVocab (const std::string& path);

    // BPE merge rules in order (earlier = higher priority)
    std::vector<std::pair<std::string,std::string>> merges_;
    // merge -> rank lookup for O(1) query
    std::map<std::pair<std::string,std::string>, int> mergeRank_;
    // token string -> id
    std::unordered_map<std::string,int> vocab_;
};

}  // namespace Sam3
