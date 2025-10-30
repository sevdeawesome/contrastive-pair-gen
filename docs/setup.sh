# SETUP CLAUDE CODE

curl -fsSL https://get.pnpm.io/install.sh | sh -
source ~/.bashrc

pnpm env use --global lts


pnpm install -g @anthropic-ai/claude-code


git config --global user.name "sevdeawesome"
git config --global user.email "sevdeawesome@gmail.com"



#REQUIRED UTILS
sudo apt-get install jq ripgrep



# MAKE SURE TO INSTALL THE REQUIREMENTS IN REQUIREMENTS.TXT