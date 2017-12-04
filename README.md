## Detecting-Roads-in-Aerial-Imagery

In the geospatial arena, machine learning focuses on the application of big data analytics to automate the extraction of specific information from geospatial data sets. The most common are imagery by airplane, UAV or satellite, which traditionally are analyzed manually to identify features, land use/land cover and changing conditions on the ground.

Supervised machine learning requires ingestion of a sample data set covering a small geographic area to ‘train’ the algorithms to identify specific features or ground conditions, such as building rooftops or road networks. The machine learning platform then scales up its big data analytics capabilities to search much larger regional or even global databases of imagery to find other instances of those features.

The benefit of geospatial machine learning is that every pixel is analyzed and the information is extracted faster than would be possible with manual methods.


**Data resources, what it looks like, and what kind of preprocessing to do:**

* City of Boulder (https://bouldercolorado.gov/open-data/boulder-street-centerlines/)
* NAIP (https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/index)
* Data will consist of Ortho-rectified aerial imagery (GeoTiff, jpeg), Street Centerlines in line vector format (shapefile, kml, GeoJson, etc.) for determining binary classification.
* Imagery must be pan-sharpened, radiometrically consistent, cloud-free, etc. Street Centerlines must overlay accurately on imagery.

 **High level description of analysis:**

* Combine machine learning algorithms and GIS applications to automate feature classification.

 **Tools to use:**

* Deep learning with a Convolutional Neural Network (CNN)
* Keras / TensorFlow backend
* AWS

**Resources (White papers, Git pages, Tutorials, etc.):**

* Detecting population centers in Nigeria - http://gbdxstories.digitalglobe.com/building-detection/
* Pool Detection Using Deep Learning https://github.com/DigitalGlobe/mltools/tree/master/examples/polygon_classify_cnn
* The Keras blog - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* Machines that can see: Convolutional Neural Networks - https://shafeentejani.github.io/2016-12-20/convolutional-neural-nets/
