% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classifySequence(todo).R
\name{classifySequence}
\alias{classifySequence}
\title{Leverage sequences to classify images}
\usage{
classifySequence(animals, mlpredictions, classfile, outfile = NA, maxdiff = 60)
}
\arguments{
\item{animals}{subselection of all images that contain MD animals}

\item{mlpredictions}{classifier predictions}

\item{classfile}{list of all possible classes}

\item{outfile}{file to which results will be saved}

\item{maxdiff}{maximum difference between images in seconds to be included in a sequence, defaults to 60}
}
\value{
reclassified animals dataframe
}
\description{
Images must be within maxdiff seconds of each other to be considered in sequence.
This function retains "Empty" classification even if other images within the
sequence are predicted to contain animals.
Classification confidence is weighted by MD confidence.
}
\examples{
\dontrun{
predictions <-classifyCropsSpecies(images,modelfile,resize=456)
animals <- classifySequences(images,predictions,classes,17,maxdiff=60)
}
}
