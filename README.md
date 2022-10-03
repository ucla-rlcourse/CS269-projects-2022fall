# UCLA CS269 2022 Fall Reinforcement Learning Course Project

Project page: https://ucla-rlcourse.github.io/CS269-projects-2022fall/

## Instruction for running this site locally

1. Follow the first 2 steps in [pull-request-instruction](pull-request-instruction.md)

2. Installing Ruby with version 3.0.0 if you are using a Mac, and ruby 2.7 should work for Linux, check https://www.ruby-lang.org/en/documentation/installation/ for instruction.

3. Installing Bundler and jekyll with
```
gem install --user-install bundler jekyll
# Make sure you are in project-2022fall folder.
bundler install
bundle add webrick
```

4. Run your site with
```
bundle exec jekyll serve
```
You should see an address pop on the terminal (http://127.0.0.1:4000/CS269-projects-2022fall
/ by default), go to this address with your browser.

Resources:

* Install Jekyll on MacOS:  https://jekyllrb.com/docs/installation/macos/



## Working on the project

1. Create a folder with your team id under ```./assets/images/your-teamid```, you will use this folder to store all the images in your project.

2. Copy the template at [`./_posts/2022-10-03-team00-instruction-to-post.md`](./_posts/2022-10-03-team00-instruction-to-post.md) and rename it with format "year-month-date-yourteamid-projectshortname.md" under ```./_posts/```, for example, **2022-10-03-team00-instruction-to-post.md**

3. Check out the [sample posts](https://ucla-rlcourse.github.io/CS269-projects-2022fall/) we provide and the [source code](_posts/2022-10-03-team00-instruction-to-post.md) as well as [basic Markdown syntax](https://www.markdownguide.org/basic-syntax/).

4. Start your work in your .md file. You may only edit the .md file you just copied and renamed, and add images to ```./assets/images/your-teamid```. Please do **NOT** change any other files in this repo.

Once you save the .md file, jekyll will synchronize the site and you can check the changes on browser.

## Submission
We will use git pull request to manage submissions.

Once you've done, follow steps 3 and 4 in [pull-request-instruction](pull-request-instruction.md) to make a pull request BEFORE the deadline. Please make sure not to modify any file except your .md file and your images folder. We will merge the request after all submissions are received, and you should able to check your work in the project page on next week of each deadline.
