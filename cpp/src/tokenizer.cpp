// tokenizer.cpp
// Author: Jason Hughes
// Date:   2026
//
// CLIP BPE tokenizer implementation for SAM3.
// Identical algorithm to jhughes50/clipper CLIPTokenizer.

#include "sam3/tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace Sam3
{

CLIPTokenizer::CLIPTokenizer(const std::string& merges_path,
                             const std::string& vocab_path)
{
    loadMerges(merges_path);
    loadVocab(vocab_path);
}

// ---------------------------------------------------------------------------

void CLIPTokenizer::loadMerges(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("CLIPTokenizer: cannot open merges file: " + path);

    std::string line;
    bool first = true;
    int rank = 0;
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        if (first) { first = false; continue; }  // skip "#version" header

        std::istringstream ss(line);
        std::string a, b;
        if (!(ss >> a >> b)) continue;

        merges_.emplace_back(a, b);
        mergeRank_[{a, b}] = rank++;
    }
}

void CLIPTokenizer::loadVocab(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("CLIPTokenizer: cannot open vocab file: " + path);

    Json::Value  root;
    Json::Reader reader;
    if (!reader.parse(f, root))
        throw std::runtime_error("CLIPTokenizer: failed to parse vocab JSON.");

    for (const auto& key : root.getMemberNames())
        vocab_[key] = root[key].asInt();
}

// ---------------------------------------------------------------------------

std::vector<int> CLIPTokenizer::tokenize(const std::string& text) const
{
    // 1. Lowercase
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // 2. Split into words by whitespace
    std::vector<std::string> words;
    {
        std::istringstream ss(lower);
        std::string w;
        while (ss >> w) words.push_back(w);
    }

    // 3. Convert each word into a sequence of byte-level tokens,
    //    appending </w> to the last character of each word.
    std::vector<std::string> tokens;
    tokens.push_back("<|startoftext|>");

    for (const auto& word : words)
    {
        std::vector<std::string> chars;
        for (std::size_t i = 0; i < word.size(); ++i)
        {
            std::string ch(1, word[i]);
            if (i + 1 == word.size())
                ch += "</w>";
            chars.push_back(ch);
        }

        // 4. BPE: iteratively apply the highest-priority merge rule
        while (chars.size() > 1)
        {
            int    best_rank = std::numeric_limits<int>::max();
            size_t best_idx  = 0;
            bool   found     = false;

            for (size_t i = 0; i + 1 < chars.size(); ++i)
            {
                auto it = mergeRank_.find({chars[i], chars[i + 1]});
                if (it != mergeRank_.end() && it->second < best_rank)
                {
                    best_rank = it->second;
                    best_idx  = i;
                    found     = true;
                }
            }

            if (!found) break;

            // Apply the merge
            std::string merged = chars[best_idx] + chars[best_idx + 1];
            chars.erase(chars.begin() + static_cast<int>(best_idx) + 1);
            chars[best_idx] = merged;
        }

        // 5. Map each BPE token to its vocabulary id
        for (const auto& tok : chars)
        {
            auto it = vocab_.find(tok);
            if (it != vocab_.end())
                tokens.push_back(tok);
            // unknown tokens are silently skipped (CLIP vocab is nearly complete)
        }
    }

    tokens.push_back("<|endoftext|>");

    // 6. Convert token strings to IDs
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens)
    {
        auto it = vocab_.find(tok);
        ids.push_back(it != vocab_.end() ? it->second : getPaddingToken());
    }

    return ids;
}

// ---------------------------------------------------------------------------

const std::vector<std::pair<std::string,std::string>>&
CLIPTokenizer::getMerges() const { return merges_; }

const std::unordered_map<std::string,int>&
CLIPTokenizer::getVocab()  const { return vocab_; }

}  // namespace Sam3
