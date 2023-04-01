.. _using_docker:

Moabb and docker
================

If you want to use the docker image pre-build
---------------------------------------------

Moabb has a default image to run the benchmark. You have two options to
download this image: build from scratch or pull from the docker hub.
**We recommend pulling from the docker hub**.

If this were your first time using docker, you would need to **install
the docker** and **login** on docker hub. We recommend the
`official <https://docs.docker.com/desktop/install/linux-install/>`__
docker documentation for this step, it is essential to follow the
instructions.

After installing docker, you can pull the image from the docker hub:

.. code:: bash

   docker pull baristimunha/moabb
   # rename the tag to moabb
   docker tag baristimunha/moabb moabb

If you want to build the image from scratch, you can use the following
command at the root. You may have to login with the API key in the `NGC
Catalog <https://catalog.ngc.nvidia.com/>`__ to run this command.

.. code:: bash

   bash docker/create_docker.sh

With the image downloaded or rebuilt from scratch, you will have an
image called ``moabb``. To run the default benchmark, still at the root
of the project, and you can use the following command:

.. code:: bash

   mkdir dataset
   mkdir results
   mkdir output
   bash docker/run_docker.sh PATH_TO_ROOT_FOLDER

An example of the command is:

.. code:: bash

   cd /home/user/project/moabb
   mkdir dataset
   mkdir results
   mkdir output
   bash docker/run_docker.sh /home/user/project/moabb

Note: It is important to use an absolute path for the root folder to
run, but you can modify the run_docker.sh script to save in another path
beyond the root of the project. By default, the script will save the
results in the project’s root in the folder ``results``, the datasets in
the folder ``dataset`` and the output in the folder ``output``.
