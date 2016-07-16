from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair


cdef extern from "split.h":
    # split a string on a single delimiter character (delim)
    vector[string]& split(const string &s, char delim, vector[string] &elems)
    vector[string]  split(const string &s, char delim)
    # split a string on any character found in the string of delimiters (delims)
    vector[string]& split(const string &s, const string& delims, vector[string] &elems)
    vector[string]  split(const string &s, const string& delims)


cdef extern from "Variant.h" namespace "vcf":

    cdef enum VariantFieldType:
        FIELD_FLOAT = 0
        FIELD_INTEGER
        FIELD_BOOL
        FIELD_STRING
        FIELD_UNKNOWN
    
    cdef enum VariantFieldNumber:
        ALLELE_NUMBER = -2
        GENOTYPE_NUMBER = -1
    
    const int INDEX_NONE = -1
    const int NULL_ALLELE = -1
    
    VariantFieldType typeStrToFieldType(string& typeStr)

    cdef cppclass VariantCallFile:
#        istream* file
#        Tabix* tabixFile
        bool usingTabix
        string header
        string line
        string fileformat
        string fileDate
        string source
        string reference
        string phasing
        map[string, VariantFieldType] infoTypes
        map[string, int] infoCounts
        map[string, VariantFieldType] formatTypes
        map[string, int] formatCounts
        vector[string] sampleNames
        bool parseInfo
        bool parseSamples
        bool _done
        void updateSamples(vector[string]& newSampleNames) except +
        void addHeaderLine(string line) except +
        void removeInfoHeaderLine(string line) except +
        void removeGenoHeaderLine(string line) except +
        vector[string] infoIds() except +
        vector[string] formatIds() except +
        vector[string] filterIds() except +
        bool open(string& filename) except +
        bool openFile(string& filename) except +
        bool openTabix(string& filename) except +
#        bool open(istream& stream) except +
#        bool open(ifstream& stream) except +
        bool openForOutput(string& headerStr) except +
        bool is_open() except +
        bool eof() except +
        bool done() except +
        bool parseHeader(string& headerStr) except +
        bool parseHeader() except +
        bool getNextVariant(Variant& var) except +
        bool setRegion(string region) except +
        bool setRegion(string seq, long int start, long int end) except +


    cdef cppclass VariantAllele:
        string ref
        string alt
        string repr
        long position
        VariantAllele(string r, string a, long p) except +
        
    
    cdef cppclass Variant:
        string sequenceName
        long position
        string id
        string ref
        vector[string] alt
        vector[string] alleles
        map[string, int] altAlleleIndexes
        map[string, vector[VariantAllele] ] parsedAlternates(bool includePreviousBaseForIndels,
                                 bool useMNPs,
                                 bool useEntropy,
                                 float matchScore,
                                 float mismatchScore,
                                 float gapOpenPenalty,
                                 float gapExtendPenalty,
                                 float repeatGapExtendPenalty,
                                 string flankingRefLeft,
                                 string flankingRefRight) except +
        map[string, string] extendedAlternates(long int newPosition, long int length) except +
        string originalLine
        string filter
        double quality
        VariantFieldType infoType(string& key) except +
        map[string, vector[string]] info
        map[string, bool] infoFlags
        VariantFieldType formatType(string& key) except +
        vector[string] format
        map[string, map[string, vector[string]]] samples
        vector[string] sampleNames
        vector[string] outputSampleNames
        VariantCallFile* vcf
        void removeAlt(string& altallele) except +
        Variant() except +
        Variant(VariantCallFile& v) except +
        void setVariantCallFile(VariantCallFile& v) except +
        void setVariantCallFile(VariantCallFile* v) except +
        void parse(string& line, bool parseInfo, bool parseSamples) except +
        void addFilter(string& tag) except +
        bool getValueBool(string& key, string& sample, int index) except +
        double getValueFloat(string& key, string& sample, int index) except +
        string getValueString(string& key, string& sample, int index) except +
        bool getSampleValueBool(string& key, string& sample, int index) except +
        double getSampleValueFloat(string& key, string& sample, int index) except +
        string getSampleValueString(string& key, string& sample, int index) except +
        bool getInfoValueBool(string& key, int index) except +
        double getInfoValueFloat(string& key, int index) except +
        string getInfoValueString(string& key, int index) except +
#        void printAlt(ostream& out)
#        void printAlleles(ostream& out)
        int getAltAlleleIndex(string& allele) except +
        void updateAlleleIndexes() except +
        void addFormatField(string& key) except +
        void setOutputSampleNames(vector[string]& outputSamples) except +
        map[pair[int, int], int] getGenotypeIndexesDiploid() except +
        int getNumSamples() except +
        int getNumValidGenotypes() except +
        

cdef class PyVariantCallFile:

    cdef VariantCallFile *thisptr
