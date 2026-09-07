rm -r dist brawlgym_ppo.egg-info
python -m build && twine check dist/* && twine upload dist/*
rm -r dist brawlgym_ppo.egg-info
