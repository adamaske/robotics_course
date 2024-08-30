echo installing pytohn dependencies from ./requirements.txt

call venv\Scripts\Activate
call python -m pip install -r requirements.txt
call venv\Scripts\Deactivate

echo requirements installed... 
exit
