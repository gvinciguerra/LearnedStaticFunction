#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <valarray>
#include <vector>

namespace learnedretrieval {

/** Reads a learned retrieval dataset from a CSV file */
class DatasetReader {
    std::vector<std::string> classes;
    std::fstream file;
    std::string buffer;
    std::string field;

public:

    struct Row {
        std::string example;                ///< The example string
        uint32_t label;                     ///< The true label of the example
        std::valarray<float> probabilities; ///< The predicted class probabilities
    };

    DatasetReader() = default;

    /** Creates a new dataset reader from the given file. */
    DatasetReader(const std::string &path) : classes(), file(path), buffer(), field() {
        if (!file.is_open())
            throw std::runtime_error("Could not open file " + path);

        // Parse classes from header columns
        std::string line;
        std::getline(file, line);
        size_t pos = 0, prev_pos = 0;
        while ((pos = line.find(',', pos + 1)) != std::string::npos) {
            if (pos > 0 && line[pos - 1] != '\\')
                classes.push_back(line.substr(prev_pos, pos - prev_pos));
            prev_pos = pos + 1;
        }
        if (line[pos - 1] != '\\')
            classes.push_back(line.substr(prev_pos));

        if (classes[0] != "example" || classes[1] != "label")
            throw std::runtime_error("Invalid header columns");
        classes.erase(classes.begin(), classes.begin() + 2);
    }

    /** Reads the next row from the dataset. Returns false if there are no more rows. */
    bool next_row(Row &row) {
        if (file.peek() == std::ifstream::traits_type::eof())
            return false;

        advance_example();
        if (field.empty() && file.eof())
            return false;

        row.example = field;

        advance_field();
        row.label = std::stoi(field);

        if (row.probabilities.size() != classes.size())
            row.probabilities.resize(classes.size());

        for (size_t i = 0; i < classes.size() - 1; i++) {
            advance_field();
            row.probabilities[i] = std::stof(field);
        }
        advance_line();
        row.probabilities[classes.size() - 1] = std::stof(field);

        if (std::fabs(row.probabilities.sum() - 1.) > 1e-4)
            throw std::runtime_error("Probabilities do not sum to 1");

        return true;
    }

    /** Returns the number of distinct classes in the dataset. */
    size_t num_classes() { return classes.size(); }

    /** Returns the name of the dataset class at the given index. */
    std::string class_name(size_t i) { return classes[i]; }

private:

    void advance_example() {
        if (file.peek() != '"') {
            std::getline(file, field, ',');
            return;
        }

        file.get(); // consume '"'
        field.clear();
        while (true) { // handle double "" escape
            std::getline(file, buffer, '"');
            field += buffer;
            if (file.peek() == '"') {
                field.push_back(char(file.get())); // add escaped "
            } else {
                if (file.get() != ',')
                    throw std::runtime_error("Expected ',' after example");
                break;
            }
        }
    }

    void advance_field() { std::getline(file, field, ','); }

    void advance_line() { std::getline(file, field); }
};

}