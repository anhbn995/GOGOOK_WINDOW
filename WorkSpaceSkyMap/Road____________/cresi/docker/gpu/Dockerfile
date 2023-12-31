# nvidia-docker build -t cresi_v3_image .
# nvidia-docker run -it --ipc=host -v /local_data:/local_data/ -p 9111:9111 --name cresi_v3 cresi_v3_image
# To use jupyter notebook, inside container run:
# jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=9111
# back on local machine, invoke the following in a web browser: https://localhost:9111

FROM nvidia/cuda:9.2-devel-ubuntu16.04
# FROM nvidia/cuda:10.0-cudnn7-devel
MAINTAINER avanetten

ENV CUDNN_VERSION 7.3.0.29
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
	apt-utils \
	bc \
	bzip2 \
	ca-certificates \
	curl \
	git \
	libgdal-dev \
	libssl-dev \
	libffi-dev \
	libncurses-dev \
	libgl1 \
	jq \
	nfs-common \
	parallel \
	python-dev \
	python-pip \
	python-wheel \
	python-setuptools \
	unzip \
	vim \
	tmux \
	wget \
	build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
# RUN ipython3 kernel install

# use conda-forge instead of default channel
RUN conda update conda && \
    #conda config --remove channels defaults && \
    conda config --add channels conda-forge

# set up conda environment and add to $PATH
RUN conda create -n cresi python=3.6 \
                    && echo "source activate cresi" > ~/.bashrc
ENV PATH /opt/conda/envs/cresi/bin:$PATH

RUN mkdir -p /root/.torch/models
RUN wget https://download.pytorch.org/models/resnet34-333f7ec4.pth -P /root/.torch/models
RUN wget http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
RUN pip install torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl \
 	&& pip install tensorboardX \
	&& pip install torchsummary \
	&& pip install utm \
	&& pip install numba \
	&& pip install imagecodecs
			  
RUN conda install -n cresi \
		gdal=2.4.2 \
		geopandas=0.6.3 \
		fiona \
		rasterio \
		osmnx=0.12 \
		networkx=2.2 \ 
		awscli \
		affine \
		pyproj \
		pyhamcrest \
		cython \
		fiona \
		h5py \
		ncurses \
		jupyter \
		jupyterlab \
		ipykernel \
		libgdal \
		matplotlib \
		ncurses \
		numpy \
		statsmodels \
		pandas \
		pillow \
		pip \
		scipy \
		scikit-image \
		scikit-learn \
		shapely \
		rtree \
		testpath \
		tqdm \
		pandas \
		opencv \
		jupyter \
		jupyterlab \
		ipykernel \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all

RUN conda install geopandas=0.6.3 -c conda-forge
	
# update pytorch?
RUN conda install -n cresi pytorch torchvision cuda90 -c pytorch \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all
	
ENV LD_LIBRARY_PATH /miniconda/lib:${LD_LIBRARY_PATH}
RUN apt install -y libgl1-mesa-glx

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate cresi && python -m ipykernel.kernelspec

# # clone cresi
RUN git clone https://github.com/avanetten/cresi.git /opt/cresi/

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 9111

WORKDIR "/opt/cresi"
RUN ["/bin/bash"]