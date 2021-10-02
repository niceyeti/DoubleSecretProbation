#!/bin/bash

NAME="Bob"

# Variables can be dereferenced inside double quotes or not, but not in single quotes
echo "Hello $NAME!"
echo $NAME
echo "${NAME}"

# Conditional execution: 
# c1 && c2      c2 only executed if c1 succeeds (ret 0)
# c1 || c2      c2 only executed if c1 fails (ret != 0)

DNE="dkljdlksjds.com"
ping $DNE || echo "Could not ping $DNE"
HOST="localhost"
ping -c 1 $HOST && echo Successfully pinged $HOST

cat << EOF1
    I am a heredoc.
    Syntax:
        cat << MARKER
            Some multiline text, where the marker can be any unique string.
        MARKER
        If <<- is used instead then all leading tabs will be dropped.
        Variables can be dereferenced in heredocs also, like parameters:
            cat << MARKER
                Current directory is $(pwd) etc
            MARKER
        If the marker is instead single or double quoted, then no parameter substitions are made.

        The most common pattern you are likely to see is redirect to file or other output:
            cat << MARKER > file.txt
                ... some content
            MARKER

EOF1


cat << 'EOF'
Conditionals:
    Syntax:
        if [[ ]]; then
        else
        fi

    Options:
        -f: check if file exists
        -r: is readable
        -d: is directory
        -h: is symlink
        -w: is writable
        -z: check if string is empty
        -eq: check for equality
        if [[ -f $SOMEPATH ]]; then ..
        fi


        [[ a && b ]] logical AND
        [[ a || b ]] logical OR
        [[ ! a ]]    logical NOT

        [[ -z STRING]] empty string check
        [[ -n STRING ]] not empty string
        [[ STRING == STRING ]] strings equal
        [[ STRING != STRING ]] strings not equal
        [[NUM -eq NUM]] numbrs equal
        [[NUM -ne NUM]] numbrs not equal
        [[NUM -lt NUM]] less than
        [[NUM -gt NUM]] greater than
        [[NUM -ge NUM]] greater than or equal to
        [[NUM -le NUM]] less than or equal to
        ((a < b)) numeric conditionals (probably preferable for readability)

        if ping -c 1 nonexistent_domain.com; then
            echo ping succeeded, domain is reachable....
        else
            echo ping failed, I really cant help ye...
        fi
EOF


if [[ -z "" ]]; then
    echo String is empty
fi

if [[ "abc" -eq "abc" ]]; then
    echo "Strings are equal"
else
    echo "Unreachable"
fi

if [[ -f "./main.go" ]]; then
    echo Found main.go
else
    echo "Could not find main.go"
fi

cat << BRACE_EXPANSION
    Syntax for brace expansion:
    {1..5}  is same as 1 2 3 4 5
    {A,B}.js is same as A.Js B.js
    {A,B} is same as A, B
BRACE_EXPANSION

cat << 'STRING_SLICING'
    Syntax for string slicing: for the most part, just look up as needed.
        name="Jesse"
        echo "${name}"
        echo ${name:0:1}
STRING_SLICING

cat << 'PARAMETER_EXPANSION'
    Syntax for parameter expansion:
        somepath="/code/myprog.cpp"
        echo ${somepath%}.cpp    # output: '/code/myprog'

        ${FOO%suffix}  removes 'suffix'
        ${FOO#prefix}  removes 'prefix'

        A few other options available: replace all, first match, etc
PARAMETER_EXPANSION

cat << COMMENTS
    Syntax for comments: 
        # single line comment
        : '
            a
            multiline 
            comment
        '
COMMENTS

cat << LOOPS
    Syntax for for-loops:
        for i in {1..5}; do
            echo $i is neat
        done

        for path in /bin/*; do

        done

        A C-like for-loop:
            for((i = 0; i < 10; i++)); do
                echo $i ...
            done

        Forever loop:
            while true; do
                echo Do stuff here...
            done

            while read line; do
                echo $line
            done
LOOPS

for i in {1..5}; do
    echo $i is neat
done

for((i = 20; i < 25; i++)); do
    echo $i is neater
done

echo
echo "A bunch of paths:"
for i in /etc/ssh/*; do
    echo $i
done

echo "Enter some text: "
while read line; do
    if [[ $line == "quit" ]]; then
        break
    fi

    echo "Your text was: $line"
    echo "Enter some text (enter 'quit' to leave loop): "
done

cat << REDIRECTION
    echo foo > bar.txt
    echo foo again >> bar.txt
    echo some errors 2> error.log   # output stderr to file
    ./some_command 2>&1          # redirect stderr of some_command to stdout
    ./some_command 2>/dev/null   # dump error output
REDIRECTION

cat << PRINTF
    printf command is straightforward mapping to c-like printf():
        printf "%s, your balance is %f" Bob 200
PRINTF


cat << SPECIAL_VARS
    $* all params as a single word
    $# the number of args
    $@ 
    $0 
    $^ the hell if I know

    $$ pid of current process
    $? exit status of last task
    $! pid of last background task (eg. 'sleep 99999 &;')
SPECIAL_VARS

cat << 'FUNCTIONS'
    foo() {
        arg1=$0
        arg2=$1
        ...

        echo "Some output, so this func can be called/processed as result=$(foo)
    }
FUNCTIONS

