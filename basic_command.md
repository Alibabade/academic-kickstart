1. add a new post:
hugo new --kind post content/post/post_name.md
   add a new project:
hugo new --kind project content/project/project_name    
   remove some commits
git reset --hard HEAD~$num$ #num is how many commits you want to delete

More details can be found：
https://wowchemy.com/docs/

2. open server for sites
hugo server
3. git status
4. git add .
5. git commit -m info
6. git push
7. git clone
8. git merge		#merge two branches you are working on
9. git init
10. git branch    #list out all the branches.
11. git checkout 	#switch to different branches
12. git reset		#set index to the latest commit that you want to work on with
13. git rebase    #operates like merge, it is fine rebase your own local branches but don't rebase public branches - master
             his branch         --AA--BB              AA---BB
                                |                     |   |
                                |                     >   >  
          my branch        A---B---C    ===>  A---B---AA---BB---C


14. git pull request   #create a new branch with a commit, but the main branch won't change, if you want to add this commit to main branch,
                       #you can merge this new branch and master.
