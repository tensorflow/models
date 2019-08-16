# This file must be dot sourced from PoSh; you cannot run it directly. Do this: . ./activate.ps1

if (@($null,"Internal") -notcontains $myinvocation.commandorigin) {
    Write-Host -Foreground red "You must 'source' this script: PS> . $($myinvocation.invocationname)"
    exit 33
}

$script:THIS_PATH = $myinvocation.mycommand.path
$script:BASE_DIR = Split-Path (Resolve-Path "$THIS_PATH/..") -Parent

function global:deactivate([switch] $NonDestructive) {
    if (Test-Path variable:_OLD_VIRTUAL_PATH) {
        $env:PATH = $variable:_OLD_VIRTUAL_PATH
        Remove-Variable "_OLD_VIRTUAL_PATH" -Scope global
    }

    if (Test-Path function:_old_virtual_prompt) {
        $function:prompt = $function:_old_virtual_prompt
        Remove-Item function:\_old_virtual_prompt
    }

    if ($env:VIRTUAL_ENV) {
        Remove-Item env:VIRTUAL_ENV -ErrorAction SilentlyContinue
    }

    if (!$NonDestructive) {
        # Self destruct!
        Remove-Item function:deactivate
        Remove-Item function:pydoc
    }
}

function global:pydoc {
    python -m pydoc $args
}

# unset irrelevant variables
deactivate -nondestructive

$VIRTUAL_ENV = $BASE_DIR
$env:VIRTUAL_ENV = $VIRTUAL_ENV

New-Variable -Scope global -Name _OLD_VIRTUAL_PATH -Value $env:PATH

$env:PATH = "$env:VIRTUAL_ENV/bin:" + $env:PATH
if (!$env:VIRTUAL_ENV_DISABLE_PROMPT) {
    function global:_old_virtual_prompt {
        ""
    }
    $function:_old_virtual_prompt = $function:prompt

    if ("" -ne "") {
        function global:prompt {
            # Add the custom prefix to the existing prompt
            $previous_prompt_value = & $function:_old_virtual_prompt
            ("" + $previous_prompt_value)
        }
    }
    else {
        function global:prompt {
            # Add a prefix to the current prompt, but don't discard it.
            $previous_prompt_value = & $function:_old_virtual_prompt
            $new_prompt_value = "($( Split-Path $env:VIRTUAL_ENV -Leaf )) "
            ($new_prompt_value + $previous_prompt_value)
        }
    }
}
