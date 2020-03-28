FROM geminihub.oa.com:80/yard/env:cuda10.0-py36-env-2.1
WORKDIR /data/user/
ADD . /data/user/
RUN pip install -i http://mirror-sng.oa.com/pypi/web/simple --trusted-host mirror-sng.oa.com -r /data/user/requirements.txt
RUN python setup.py install
