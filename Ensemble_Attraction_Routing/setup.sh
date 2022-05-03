
mkdir -p inputs
mkdir -p outputs
mkdir -p maps


while getopts ":d" opt; do
  case $opt in
    d)
      wget -O gdp.zip "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"
      unzip gdp.zip
      rm gdp.zip
      rm *Metadata*

      for nam in API_NY.GDP.MKTP.CD_DS2_en_csv*.csv
      do

          newname='../data/GDP_preprocessed.csv'
          mv $nam $newname

      done

      wget -O vdem.zip "https://v-dem.net/media/datasets/Country_Year_V-Dem_Core_CSV_v12.zip"
      unzip vdem.zip
      rm vdem.zip

      rm Country_Year_V-Dem_Core_CSV_v12/*.pdf
      for nam in Country_Year_V-Dem_Core_CSV_v12/*.csv
      do

          newname='../data/vdem_preprocessed.csv'
          mv $nam $newname

      done
      rm -rf Country_Year_V-Dem_Core_CSV_v12

      wget -O pop.zip "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"
      unzip pop.zip
      rm pop.zip

      rm *Metadata*
    for nam in API_SP.POP.TOTL_DS2_en_csv*.csv
      do

          newname='../data/POP_preprocessed.csv'
          mv $nam $newname

      done
      rm -rf __MACOSX
      python setup.py
      echo "got GPD: -$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac

done
echo "Done"
