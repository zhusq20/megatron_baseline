#!/usr/bin/bash
export https_proxy=http://172.23.111.197:36556
export http_proxy=http://172.23.111.197:36556
export all_proxy=socks5://172.23.111.197:36556

# Git SSH
# https://stackoverflow.com/questions/58245255/how-to-force-git-to-use-socks-proxy-over-its-ssh-connection
# export GIT_SSH_COMMAND='ssh -o ProxyCommand="nc  172.23.111.197 36556 %h %p"'
export GIT_SSH_COMMAND='ssh -o ProxyCommand="/bin/nc -v -x 172.23.111.197:36556 %h %p"'

# Other SSH connection are not proxied

$@
