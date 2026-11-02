#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "kael/debug.h"
#include "kael/kael.h"

using kael::Tensor;

TEST(Explain, ShowsLayoutAndDtype) {
    const Tensor x = kael::zeros({2, 3}, kael::DType::F64);
    const std::string text = kael::explain(x);
    EXPECT_NE(text.find("shape: [2, 3]"), std::string::npos) << text;
    EXPECT_NE(text.find("strides: [3, 1]"), std::string::npos) << text;
    EXPECT_NE(text.find("offset: 0"), std::string::npos) << text;
    EXPECT_NE(text.find("dtype: float64"), std::string::npos) << text;
    EXPECT_NE(text.find("device: cpu"), std::string::npos) << text;
    EXPECT_NE(text.find("contiguous: yes"), std::string::npos) << text;
    EXPECT_NE(text.find("48 bytes"), std::string::npos) << text;
    EXPECT_NE(text.find("references: 1"), std::string::npos) << text;
}

TEST(Explain, ShowsSharedStorageOfViews) {
    const Tensor x = kael::zeros({4, 4});
    const Tensor view = x.T().slice(0, 1, 3);
    const std::string text = kael::explain(view);
    EXPECT_NE(text.find("contiguous: no"), std::string::npos) << text;
    EXPECT_NE(text.find("offset: 1"), std::string::npos) << text;
    EXPECT_NE(text.find("references: 2"), std::string::npos) << text;
}

TEST(Explain, ShowsAutogradState) {
    Tensor x = kael::ones({2});
    EXPECT_NE(kael::explain(x).find("requires_grad: no"), std::string::npos);
    x.requires_grad_();
    EXPECT_NE(kael::explain(x).find("leaf: yes"), std::string::npos);
    const Tensor y = x * 2;
    const std::string text = kael::explain(y);
    EXPECT_NE(text.find("requires_grad: yes"), std::string::npos) << text;
    EXPECT_NE(text.find("grad_fn: MulBackward"), std::string::npos) << text;
}

TEST(GraphDot, HasOneNodePerOperationAndOneEdgePerInput) {
    Tensor a = kael::ones({2});
    a.requires_grad_();
    const Tensor total = kael::sum(a * a + a);
    const std::string dot = kael::autograd_graph_dot(total);
    EXPECT_EQ(dot.find("digraph"), 0u) << dot;
    for (const char* name : {"SumBackward", "AddBackward", "MulBackward", "AccumulateGrad"}) {
        EXPECT_NE(dot.find(name), std::string::npos) << name << "\n" << dot;
    }
    std::size_t edges = 0;
    for (auto at = dot.find("->"); at != std::string::npos; at = dot.find("->", at + 2)) {
        ++edges;
    }
    EXPECT_EQ(edges, 5u) << dot;
    EXPECT_NE(dot.find("[2]"), std::string::npos) << dot;
}

TEST(GraphDot, ATensorWithoutGraphIsAnEmptyDigraph) {
    const std::string dot = kael::autograd_graph_dot(kael::ones({1}));
    EXPECT_EQ(dot.find("->"), std::string::npos);
}

TEST(BackwardTrace, PrintsEveryNodeWithShapeAndNorm) {
    Tensor x = kael::full({2}, 3.0);
    x.requires_grad_();
    std::ostringstream trace;
    kael::set_backward_trace(&trace);
    kael::sum(x * x).backward();
    kael::set_backward_trace(nullptr);
    const std::string text = trace.str();
    EXPECT_NE(text.find("SumBackward"), std::string::npos) << text;
    EXPECT_NE(text.find("MulBackward"), std::string::npos) << text;
    EXPECT_NE(text.find("AccumulateGrad shape=[2]"), std::string::npos) << text;
    EXPECT_NE(text.find("norm="), std::string::npos) << text;
}

TEST(BackwardTrace, IsOffByDefault) {
    Tensor x = kael::ones({1});
    x.requires_grad_();
    std::ostringstream trace;
    kael::set_backward_trace(&trace);
    kael::set_backward_trace(nullptr);
    kael::sum(x).backward();
    EXPECT_TRUE(trace.str().empty());
}

TEST(AnomalyDetection, NamesTheNodeProducingNan) {
    Tensor x = kael::full({1}, -1.0);
    x.requires_grad_();
    kael::set_anomaly_detection(true);
    try {
        kael::sum(kael::sqrt(x)).backward();
        kael::set_anomaly_detection(false);
        FAIL() << "should have thrown";
    } catch (const kael::Error& error) {
        kael::set_anomaly_detection(false);
        const std::string message = error.what();
        EXPECT_NE(message.find("Sqrt"), std::string::npos) << message;
        EXPECT_NE(message.find("NaN"), std::string::npos) << message;
    }
}

TEST(AnomalyDetection, IsOffByDefaultAndLeavesCleanGraphsAlone) {
    Tensor x = kael::full({1}, -1.0);
    x.requires_grad_();
    kael::sum(kael::sqrt(x)).backward();
    EXPECT_TRUE(std::isnan(x.grad()->item()));

    Tensor y = kael::full({1}, 4.0);
    y.requires_grad_();
    kael::set_anomaly_detection(true);
    EXPECT_NO_THROW(kael::sum(kael::sqrt(y)).backward());
    kael::set_anomaly_detection(false);
}
