#include <memory>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "kael/io/safetensors.h"
#include "kael/nn/transformer.h"
#include "kael/text/bpe.h"

#include "bind.h"

namespace kael::python {

void bind_models(nb::module_& m) {
    m.def(
        "save_safetensors",
        [](const std::string& path, const io::TensorMap& tensors, const io::Metadata& metadata) {
            io::save_safetensors(path, tensors, metadata);
        },
        nb::arg("path"), nb::arg("tensors"), nb::arg("metadata") = io::Metadata{},
        "Write tensors to a safetensors file.");
    m.def(
        "load_safetensors",
        [](const std::string& path, bool mapped) {
            return io::load_safetensors(path, mapped ? io::LoadMode::Mapped : io::LoadMode::Copy);
        },
        nb::arg("path"), nb::arg("mmap") = false,
        "Read every tensor of a safetensors file, validating its layout. With mmap the tensors "
        "share the pages of the file, and the file must not be truncated while they are alive.");
    nb::class_<text::BpeTokenizer>(m, "BpeTokenizer")
        .def(
            "__init__",
            [](text::BpeTokenizer* self, const std::string& vocabulary, const std::string& merges) {
                new (self) text::BpeTokenizer(vocabulary, merges);
            },
            nb::arg("vocabulary"), nb::arg("merges"))
        .def(
            "encode",
            [](const text::BpeTokenizer& tokenizer, const std::string& input) {
                return tokenizer.encode(input);
            },
            nb::arg("text"), nb::call_guard<nb::gil_scoped_release>())
        .def(
            "decode",
            [](const text::BpeTokenizer& tokenizer, const std::vector<std::int64_t>& ids) {
                const std::string decoded = tokenizer.decode(ids);
                return nb::bytes(decoded.data(), decoded.size());
            },
            nb::arg("ids"))
        .def_prop_ro("vocabulary_size", &text::BpeTokenizer::vocabulary_size);
    m.def("read_safetensors_metadata",
          [](const std::string& path) { return io::read_safetensors(path).metadata; });

    const auto as_tuple = [](const SortResult& result) {
        return nb::make_tuple(result.values, result.indices);
    };
    m.def("cumsum", &kael::cumsum, nb::arg("input"), nb::arg("dim"));
    m.def(
        "sort",
        [as_tuple](const Tensor& t, std::int64_t dim, bool descending) {
            return as_tuple(kael::sort(t, dim, descending));
        },
        nb::arg("input"), nb::arg("dim") = -1, nb::arg("descending") = false);
    m.def(
        "topk",
        [as_tuple](const Tensor& t, std::int64_t k, std::int64_t dim, bool largest) {
            return as_tuple(kael::topk(t, k, dim, largest));
        },
        nb::arg("input"), nb::arg("k"), nb::arg("dim") = -1, nb::arg("largest") = true);
    m.def("tril", &kael::tril, nb::arg("input"), nb::arg("diagonal") = 0);
    m.def("triu", &kael::triu, nb::arg("input"), nb::arg("diagonal") = 0);
    m.def("multinomial", &kael::multinomial, nb::arg("weights"), nb::arg("samples"),
          nb::arg("replacement") = false);

    m.def("explain", &kael::explain, nb::arg("tensor"), "Describe the memory layout of a tensor.");
    m.def("autograd_graph", &kael::autograd_graph_dot, nb::arg("tensor"),
          "The graph behind a tensor, in Graphviz DOT format.");
    m.def("set_anomaly_detection", &kael::set_anomaly_detection,
          "Make backward fail at the first node producing NaN.");

    nb::class_<nn::GPT>(m, "GPT", "A GPT-2 style transformer implemented in C++.")
        .def(
            "__init__",
            [](nn::GPT* self, std::int64_t vocab_size, std::int64_t max_positions,
               std::int64_t layers, std::int64_t heads, std::int64_t dimension, double dropout,
               bool tanh_gelu, DType dtype) {
                new (self) nn::GPT(nn::GptConfig{vocab_size, max_positions, layers, heads,
                                                 dimension, dropout, tanh_gelu, dtype});
            },
            nb::arg("vocab_size") = 50257, nb::arg("max_positions") = 1024, nb::arg("layers") = 12,
            nb::arg("heads") = 12, nb::arg("dimension") = 768, nb::arg("dropout") = 0.0,
            nb::arg("tanh_gelu") = true, nb::arg("dtype") = DType::F32)
        .def(
            "__call__",
            [](nn::GPT& model, const Tensor& indices) { return model.forward(indices); },
            nb::call_guard<nb::gil_scoped_release>())
        .def("parameters", &nn::Module::parameters)
        .def("named_parameters", &nn::Module::named_parameters)
        .def("num_parameters", &nn::Module::num_parameters)
        .def("train", &nn::Module::train, nb::arg("training") = true)
        .def("eval", &nn::Module::eval)
        .def("zero_grad", &nn::Module::zero_grad)
        .def("to",
             [](nn::GPT& model, const std::string& device) { model.to(parse_device(device)); })
        .def("state_dict", &nn::Module::state_dict)
        .def("load_state_dict", &nn::Module::load_state_dict)
        .def(
            "load_gpt2",
            [](nn::GPT& model, const std::string& path) { nn::load_gpt2_weights(model, path); },
            "Load GPT-2 weights from a HuggingFace safetensors file.")
        .def(
            "generate",
            [](nn::GPT& model, const Tensor& prompt, std::int64_t max_new_tokens,
               double temperature, std::int64_t top_k, bool use_cache,
               const std::optional<nb::callable>& on_token) {
                nn::GenerateOptions options{.max_new_tokens = max_new_tokens,
                                            .temperature = temperature,
                                            .top_k = top_k,
                                            .use_cache = use_cache};
                if (on_token) {
                    options.on_token = [&on_token](const Tensor& token) {
                        const nb::gil_scoped_acquire gil;
                        const nb::object answer = (*on_token)(token);
                        if (answer.is_none()) {
                            return true;
                        }
                        const int truth = PyObject_IsTrue(answer.ptr());
                        if (truth < 0) {
                            nb::raise_python_error();
                        }
                        return truth == 1;
                    };
                }
                return nn::generate(model, prompt, options);
            },
            nb::arg("prompt"), nb::arg("max_new_tokens") = 16, nb::arg("temperature") = 1.0,
            nb::arg("top_k") = 0, nb::arg("use_cache") = true, nb::arg("on_token") = nb::none(),
            nb::call_guard<nb::gil_scoped_release>());
}

} // namespace kael::python
