lazy val commonSettings = Seq(
  name := "reddit-prediction",
  version := "1.0",
  scalaVersion := "2.10.4"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    crossPaths := false,

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "1.5.2" % "provided",
      "org.apache.spark" %% "spark-sql" % "1.5.2" % "provided",
      "org.apache.spark" %% "spark-mllib" % "1.5.2" % "provided",

      "joda-time" % "joda-time" % "2.9.1",
      "com.github.scopt" %% "scopt" % "3.3.0",

      // BigQuery Connector
      "com.google.cloud.bigdataoss" % "bigquery-connector-parent" % "0.7.3",
      "com.google.cloud.bigdataoss" % "bigquery-connector" % "0.7.3-hadoop2"
      // "com.google.apis" % "google-api-services-bigquery" % "v2-rev248-1.21.0",
      // "com.google.code.gson" % "gson" % "2.5",
      // "com.google.cloud.bigdataoss" %% "bigquery-connector" % "0.7.3-hadoop2" from "https://oss.sonatype.org/content/repositories/staging/com/google/cloud/bigdataoss/bigquery-connector/0.7.3-hadoop2/bigquery-connector-0.7.3-hadoop2.jar"
    )

    // unmanagedJars in Compile += file("lib/gcs-connector-1.4.3-hadoop2.jar")
    // unmanagedJars in Compile += file("lib/bigquery-connector-latest.jar")
  )
