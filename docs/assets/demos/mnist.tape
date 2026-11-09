# Records an MLP training on MNIST on the CPU. Run from the repository root:
#   vhs docs/assets/demos/mnist.tape

Output docs/assets/demos/mnist.gif
Set Height 640
Source docs/assets/demos/settings.tape

Sleep 500ms
Type "build-release/examples/mnist_mlp data/mnist"
Sleep 600ms
Enter
Wait+Line@60s /kael\.cpp \$\s*$/
Sleep 4s
