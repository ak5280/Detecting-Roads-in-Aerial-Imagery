## Detecting-Roads-in-Aerial-Imagery

In the geospatial arena, machine learning focuses on the application of big data analytics to automate the extraction of specific information from geospatial data sets. The most common are imagery by airplane, UAV or satellite, which traditionally are analyzed manually to identify features, land use/land cover and changing conditions on the ground.

Supervised machine learning requires ingestion of a sample data set covering a small geographic area to ‘train’ the algorithms to identify specific features or ground conditions, such as building rooftops. The machine learning platform then scales up its big data analytics capabilities to search much larger regional or even global databases of imagery to find other instances of those features.

The benefit of geospatial machine learning is that every pixel is analyzed and the information is extracted faster than would be possible with manual methods.


**Data resources, what it looks like, and what kind of preprocessing to do:**

* City of Boulder (https://bouldercolorado.gov/open-data/city-of-boulder-building-footprints/)
* City of Bloomington, IN (https://catalog.data.gov/dataset/building-footprint-gis-data)
* East View Geospatial (https://www.geospatialworld.net/news/east-view-geospatial-announces-training-data-library-geospatial-machine-learning/)
* Geoscape (https://www.geoscape.com.au/get-geoscape/)
* Data will consist of Ortho-rectified Earth imagery (GeoTiff, jp2, ecw, etc.), Building footprints in polygon vector format (shapefile, kml, Geojson, etc.) for training
* Imagery must be pan-sharpened, radiometrically consistent, cloud-free, etc. Building footprints must overlay accurately on imagery.

 **High level description of analysis:**

* Transform Earth imagery into vector feature maps
* Combine machine learning algorithms and remote sensing techniques and to automate feature extraction
* Provide a digital representation of each building, comprising a two-dimensional building footprint

 **Tools to use:**

* Deep learning with a Convolutional Neural Network (CNN)
* EC2

 **Plans for presentation results:**

* A nice README along with a Jupyter Notebook for code description
* It might also be nice to build a blog/webpage too for future presentations/interviews

**High level timeline for project stages:**

* Start gathering data ASAP
* Begin investigating how to build CNNs (now) Nov 6 - Nov 20
* See what possibilities there are for outputting vector components Nov 20 - Nov 27
* Code freeze Nov 28
* Prepare presentation Nov 28 - Dec 3

**Resources (White papers, Git pages, Tutorials, etc.):**

* Detecting population centers in Nigeria - http://gbdxstories.digitalglobe.com/building-detection/
* Pool Detection Using Deep Learning - https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_cnn
* Use Machine Learning to Create Building Heights in OSM - https://2017.stateofthemap.us/program/use-machine-learning-to-create-building-heights.html
