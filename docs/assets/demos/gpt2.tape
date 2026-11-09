# Records GPT-2 small generating text on the CPU. Run from the repository root:
#   vhs docs/assets/demos/gpt2.tape

Output docs/assets/demos/gpt2.gif
Set Height 400
Source docs/assets/demos/settings.tape

Sleep 500ms
Type `build-release/examples/gpt2_generate "The meaning of life is" --tokens 60`
Sleep 600ms
Enter
Wait+Line@60s /kael\.cpp \$\s*$/
Sleep 4s
