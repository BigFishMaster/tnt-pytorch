FROM geminihub.oa.com:80/yard/env:cuda10.0-py36-env-2.1
RUN git clone https://github.com/BigFishMaster/tnt.git
RUN cd tnt
RUN pip install -r requirements.txt
RUN python setup.py install
