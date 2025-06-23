
For Ocean View repo:
conda activate oceanview

repo inside repo

In inside repo, the workflow is to add as submodule is first commit and push the folder into git page and then remove from local and git clone as submodule. see below codes:

from the sub-repo
echo "# test" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/akashspunnayil/test.git
git push -u origin main
git pull

	from  the parent repo
	cp Face_Detection ../
	rm -rf Face_Detection
	git rm -rf Face_Detection
	cp ../Face_Detection/* Face_Detection/
	git submodule add https://github.com/akashspunnayil/Face_Detection
	

pip install pyinstaller

pyinstaller --onefile --noconsole OceanView_app_local.py \
  --exclude-module h5py \
  --exclude-module tensorflow \
  --hidden-import streamlit

