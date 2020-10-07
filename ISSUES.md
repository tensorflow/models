# If you open a GitHub issue, here is our policy.

* It must be a **bug**, a **feature request**, or a significant problem
with **documentation**.
  * Please send a pull request instead for small documentation fixes.
* The required form must be filled out.
* The issue should be related to the repository it is created in.

General help and support should be sought on [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow-model-garden) or other non-GitHub channels.

[![](https://img.shields.io/stackexchange/stackoverflow/t/tensorflow-model-garden)](https://stackoverflow.com/questions/tagged/tensorflow-model-garden)

TensorFlow developers respond to issues.
We want to focus on work that benefits the whole community such as fixing bugs
and adding new features.
It helps us to address bugs and feature requests in a timely manner.

--- 


## Reporting unresolved problems
Check our troubleshooting common issues guide and see if your issue is resolved using the steps provided.

Hopefully the troubleshooting steps above resolved your problem! If things still aren't working the way you expect them to, please let us know so that we can diagnose and hopefully fix the problem you're having.

The best way to report a bug is by providing a reproduction script. See these examples:

Git environment variables causing install to fail.
Multiple gems in a repository cannot be updated independently.
A half working script with comments for the parts you were unable to automate is still appreciated.

If you are unable to do that, please include the following information in your report:

What you're trying to accomplish
The command you ran
What you expected to happen
What actually happened
The exception backtrace(s), if any
Everything output by running bundle env
If your version of Bundler does not have the bundle env command, then please include:

Your Gemfile
Your Gemfile.lock
Your Bundler configuration settings (run bundle config)
What version of bundler you are using (run bundle -v)
What version of Ruby you are using (run ruby -v)
What version of RubyGems you are using (run gem -v)
Whether you are using RVM, and if so what version (run rvm -v)
Whether you have the rubygems-bundler gem, which can break gem executables (run gem list rubygems-bundler)
Whether you have the open_gem gem, which can cause rake activation conflicts (run gem list open_gem)
If you have either rubygems-bundler or open_gem installed, please try removing them and then following the troubleshooting steps above before opening a new ticket.

Create a gist containing all of that information, then visit the Bundler issue tracker and create a ticket describing your problem and linking to your gist.



Please understand that research models in the [research directory](https://github.com/tensorflow/models/tree/master/research)
included in this repository are experimental and research-style code.
They are not officially supported by the TensorFlow team.


