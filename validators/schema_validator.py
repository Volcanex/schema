"""
Schema.org JSON-LD Validator for Synthetic Training Data Pipeline
Three-stage validation:
1. JSON structural validity
2. Schema.org vocabulary compliance (types, properties, enumerations)
3. Factual cross-referencing against source HTML
Returns a ValidationResult with pass/fail, scores, and specific issues.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any
from html.parser import HTMLParser
# ============================================================================
# Known schema.org types and their valid properties
# We include the most common types for Irish/UK SME websites.
# This is not exhaustive — extend as needed from schema.org vocabulary.
# ============================================================================
SCHEMA_TYPES: dict[str, set[str]] = {
    # === Core types ===
    "Thing": {
        "@context", "@type", "@id", "name", "description", "url", "image",
        "sameAs", "identifier", "additionalType", "alternateName",
        "disambiguatingDescription", "mainEntityOfPage", "potentialAction",
        "subjectOf", "additionalProperty",
    },
    # === Organization hierarchy ===
    "Organization": {
        "address", "aggregateRating", "brand", "contactPoint", "department",
        "email", "employee", "event", "faxNumber", "founder", "foundingDate",
        "foundingLocation", "globalLocationNumber", "hasOfferCatalog", "iso6523Code",
        "legalName", "location", "logo", "member", "numberOfEmployees",
        "parentOrganization", "review", "slogan", "taxID", "telephone",
        "vatID", "areaServed", "award", "knowsAbout", "makesOffer",
        "ownershipFundingInfo", "publishingPrinciples", "sponsor",
    },
    "LocalBusiness": {
        "currenciesAccepted", "openingHours", "openingHoursSpecification",
        "paymentAccepted", "priceRange", "branchOf", "geo", "hasMap",
        "latitude", "longitude", "smokingAllowed", "specialOpeningHoursSpecification",
        "amenityFeature", "tourBookingPage", "containedInPlace",
    },
    # LocalBusiness subtypes — inherit all parent properties
    "Restaurant": {"servesCuisine", "acceptsReservations", "hasMenu", "menu", "starRating"},
    "Bakery": set(),
    "BarOrPub": set(),
    "CafeOrCoffeeShop": set(),
    "FastFoodRestaurant": set(),
    "IceCreamShop": set(),
    "Hotel": {"checkinTime", "checkoutTime", "numberOfRooms", "petsAllowed", "starRating", "amenityFeature"},
    "Hostel": {"checkinTime", "checkoutTime", "numberOfRooms", "petsAllowed", "starRating"},
    "Motel": {"checkinTime", "checkoutTime", "numberOfRooms", "petsAllowed", "starRating"},
    "BedAndBreakfast": {"checkinTime", "checkoutTime", "numberOfRooms", "petsAllowed", "starRating"},
    "Dentist": set(),
    "Physician": {"medicalSpecialty", "availableService", "hospitalAffiliation"},
    "Pharmacy": set(),
    "Optician": set(),
    "AutoRepair": set(),
    "AutoDealer": set(),
    "GasStation": set(),
    "HealthClub": set(),
    "DaySpa": set(),
    "BeautySalon": set(),
    "HairSalon": set(),
    "NailSalon": set(),
    "BarberShop": set(),
    "Plumber": set(),
    "Electrician": set(),
    "HVACBusiness": set(),
    "RoofingContractor": set(),
    "LockSmith": set(),
    "Locksmith": set(),
    "MovingCompany": set(),
    "HousePainter": set(),
    "GeneralContractor": set(),
    "Store": set(),
    "ClothingStore": set(),
    "ElectronicsStore": set(),
    "FurnitureStore": set(),
    "GardenStore": set(),
    "GroceryStore": set(),
    "HardwareStore": set(),
    "HobbyShop": set(),
    "HomeGoodsStore": set(),
    "JewelryStore": set(),
    "LiquorStore": set(),
    "MensClothingStore": set(),
    "MobilePhoneStore": set(),
    "MusicStore": set(),
    "OfficeEquipmentStore": set(),
    "OutletStore": set(),
    "PetStore": set(),
    "ShoeStore": set(),
    "SportingGoodsStore": set(),
    "TireShop": set(),
    "ToyStore": set(),
    "WholesaleStore": set(),
    "AccountingService": set(),
    "LegalService": set(),
    "Notary": set(),
    "Attorney": set(),
    "RealEstateAgent": set(),
    "TravelAgency": set(),
    "InsuranceAgency": set(),
    "EmploymentAgency": set(),
    "FinancialService": set(),
    "ProfessionalService": set(),
    "ChildCare": set(),
    "DryCleaningOrLaundry": set(),
    "Library": set(),
    "RecyclingCenter": set(),
    "SelfStorage": set(),
    "ShoppingCenter": set(),
    "EntertainmentBusiness": set(),
    "AmusementPark": set(),
    "Casino": set(),
    "NightClub": set(),
    "MovieTheater": {"screenCount"},
    "BowlingAlley": set(),
    "ExerciseGym": set(),
    "GolfCourse": set(),
    "SkiResort": set(),
    "SportsActivityLocation": set(),
    "StadiumOrArena": set(),
    "TennisComplex": set(),
    "GovernmentOffice": set(),
    "PostOffice": set(),
    "FireStation": set(),
    "PoliceStation": set(),
    "AnimalShelter": set(),
    "Campground": set(),
    "CivicStructure": set(),
    "ComedyClub": set(),
    "FoodEstablishment": {"servesCuisine", "acceptsReservations", "hasMenu", "menu", "starRating"},
    "LodgingBusiness": {"checkinTime", "checkoutTime", "numberOfRooms", "petsAllowed", "starRating", "amenityFeature"},
    "MedicalBusiness": set(),
    "AutomotiveBusiness": set(),
    "HealthAndBeautyBusiness": set(),
    "HomeAndConstructionBusiness": set(),
    "InternetCafe": set(),
    "RadioStation": set(),
    "TelevisionStation": set(),
    "TouristInformationCenter": set(),
    "EmergencyService": set(),
    "Hospital": {"medicalSpecialty", "availableService"},
    # === Product/Offer ===
    "Product": {
        "aggregateRating", "audience", "award", "brand", "category", "color",
        "depth", "gtin", "gtin8", "gtin12", "gtin13", "gtin14", "hasEnergyConsumptionDetails",
        "hasMerchantReturnPolicy", "height", "isAccessoryOrSparePartFor",
        "isConsumableFor", "isRelatedTo", "isSimilarTo", "itemCondition",
        "logo", "manufacturer", "material", "model", "mpn", "nsn",
        "offers", "pattern", "productID", "productionDate", "purchaseDate",
        "releaseDate", "review", "size", "sku", "slogan", "weight", "width",
        "additionalProperty", "countryOfOrigin", "countryOfAssembly",
        "countryOfLastProcessing", "hasAdultConsideration",
    },
    "Offer": {
        "acceptedPaymentMethod", "addOn", "advanceBookingRequirement",
        "aggregateRating", "areaServed", "availability", "availabilityEnds",
        "availabilityStarts", "availableAtOrFrom", "availableDeliveryMethod",
        "businessFunction", "category", "deliveryLeadTime", "eligibleCustomerType",
        "eligibleDuration", "eligibleQuantity", "eligibleRegion",
        "eligibleTransactionVolume", "gtin", "gtin8", "gtin12", "gtin13", "gtin14",
        "hasMeasurement", "hasMerchantReturnPolicy", "includesObject",
        "inventoryLevel", "itemCondition", "itemOffered", "leaseLength",
        "mpn", "offeredBy", "price", "priceCurrency", "priceSpecification",
        "priceValidUntil", "review", "seller", "serialNumber", "shippingDetails",
        "sku", "validFrom", "validThrough", "warranty",
    },
    "AggregateOffer": {
        "highPrice", "lowPrice", "offerCount", "offers",
    },
    # === Event ===
    "Event": {
        "about", "actor", "aggregateRating", "attendee", "audience", "composer",
        "contributor", "director", "doorTime", "duration", "endDate",
        "eventAttendanceMode", "eventSchedule", "eventStatus", "funder",
        "inLanguage", "isAccessibleForFree", "keywords", "location",
        "maximumAttendeeCapacity", "maximumPhysicalAttendeeCapacity",
        "maximumVirtualAttendeeCapacity", "offers", "organizer", "performer",
        "previousStartDate", "recordedIn", "remainingAttendeeCapacity",
        "review", "sponsor", "startDate", "subEvent", "superEvent",
        "translator", "typicalAgeRange", "workFeatured", "workPerformed",
    },
    "BusinessEvent": set(),
    "ChildrensEvent": set(),
    "ComedyEvent": set(),
    "CourseInstance": set(),
    "DanceEvent": set(),
    "DeliveryEvent": set(),
    "EducationEvent": set(),
    "ExhibitionEvent": set(),
    "Festival": set(),
    "FoodEvent": set(),
    "Hackathon": set(),
    "LiteraryEvent": set(),
    "MusicEvent": set(),
    "PublicationEvent": set(),
    "SaleEvent": set(),
    "ScreeningEvent": set(),
    "SocialEvent": set(),
    "SportsEvent": set(),
    "TheaterEvent": set(),
    "VisualArtsEvent": set(),
    # === Creative Works ===
    "Article": {
        "articleBody", "articleSection", "backstory", "pageEnd", "pageStart",
        "pagination", "speakable", "wordCount",
    },
    "BlogPosting": set(),
    "NewsArticle": {"dateline", "printColumn", "printEdition", "printPage", "printSection"},
    "TechArticle": {"dependencies", "proficiencyLevel"},
    "CreativeWork": {
        "about", "abstract", "accessMode", "accessModeSufficient",
        "accessibilityAPI", "accessibilityControl", "accessibilityFeature",
        "accessibilityHazard", "accessibilitySummary", "accountablePerson",
        "acquireLicensePage", "aggregateRating", "alternativeHeadline",
        "assesses", "associatedMedia", "audience", "author", "award",
        "character", "citation", "comment", "commentCount", "conditionsOfAccess",
        "contentLocation", "contentRating", "contentReferenceTime",
        "contributor", "copyrightHolder", "copyrightNotice", "copyrightYear",
        "creativeWorkStatus", "creator", "dateCreated", "dateModified",
        "datePublished", "editEIDR", "editor", "educationalAlignment",
        "educationalLevel", "educationalUse", "encoding", "encodingFormat",
        "exampleOfWork", "expires", "funder", "genre", "hasPart", "headline",
        "inLanguage", "interactionStatistic", "interactivityType",
        "interpretedAsClaim", "isAccessibleForFree", "isBasedOn", "isFamilyFriendly",
        "isPartOf", "keywords", "learningResourceType", "license",
        "locationCreated", "mainEntity", "maintainer", "material",
        "materialExtent", "mentions", "offers", "position", "producer",
        "provider", "publication", "publisher", "publisherImprint",
        "publishingPrinciples", "recordedAt", "releasedEvent", "review",
        "schemaVersion", "sdDatePublished", "sdLicense", "sdPublisher",
        "size", "sourceOrganization", "spatial", "spatialCoverage",
        "sponsor", "teaches", "temporal", "temporalCoverage", "text",
        "thumbnailUrl", "timeRequired", "translationOfWork", "translator",
        "typicalAgeRange", "usageInfo", "version", "video", "workExample",
        "workTranslation",
    },
    # === Recipe ===
    "Recipe": {
        "cookingMethod", "cookTime", "estimatedCost", "nutrition",
        "prepTime", "recipeCategory", "recipeCuisine", "recipeIngredient",
        "recipeInstructions", "recipeYield", "suitableForDiet", "totalTime",
        "tool",
    },
    # === FAQ ===
    "FAQPage": {"mainEntity"},
    "Question": {"acceptedAnswer", "answerCount", "suggestedAnswer", "eduQuestionType"},
    "Answer": {"answerExplanation", "text", "upvoteCount", "downvoteCount"},
    # === Review/Rating ===
    "Review": {
        "associatedClaimReview", "associatedMediaReview", "associatedReview",
        "itemReviewed", "negativeNotes", "positiveNotes", "reviewAspect",
        "reviewBody", "reviewRating",
    },
    "AggregateRating": {
        "bestRating", "itemReviewed", "ratingCount", "ratingValue",
        "reviewCount", "worstRating",
    },
    "Rating": {"bestRating", "ratingExplanation", "ratingValue", "worstRating"},
    # === Place/Address ===
    "Place": {
        "address", "aggregateRating", "amenityFeature", "branchCode",
        "containedInPlace", "containsPlace", "event", "faxNumber", "geo",
        "geoContains", "geoCoveredBy", "geoCovers", "geoCrosses",
        "geoDisjoint", "geoEquals", "geoIntersects", "geoOverlaps",
        "geoTouches", "geoWithin", "globalLocationNumber", "hasDriveThroughService",
        "hasMap", "isAccessibleForFree", "latitude", "logo", "longitude",
        "maximumAttendeeCapacity", "openingHoursSpecification", "photo",
        "publicAccess", "review", "slogan", "smokingAllowed",
        "specialOpeningHoursSpecification", "telephone", "tourBookingPage",
    },
    "PostalAddress": {
        "addressCountry", "addressLocality", "addressRegion", "postOfficeBoxNumber",
        "postalCode", "streetAddress",
    },
    "GeoCoordinates": {"address", "addressCountry", "elevation", "latitude", "longitude", "postalCode"},
    "VirtualLocation": {"url"},
    # === Person ===
    "Person": {
        "additionalName", "address", "affiliation", "alumniOf", "award",
        "birthDate", "birthPlace", "brand", "callSign", "children",
        "colleague", "contactPoint", "deathDate", "deathPlace", "email",
        "familyName", "faxNumber", "follows", "gender", "givenName",
        "globalLocationNumber", "hasCredential", "hasOccupation",
        "hasOfferCatalog", "hasPOS", "height", "homeLocation",
        "honorificPrefix", "honorificSuffix", "interactionStatistic",
        "isicV4", "jobTitle", "knows", "knowsAbout", "knowsLanguage",
        "memberOf", "nationality", "netWorth", "owns", "parent",
        "performerIn", "publishingPrinciples", "relatedTo", "seeks",
        "sibling", "sponsor", "spouse", "taxID", "telephone", "vatID",
        "weight", "workLocation", "worksFor",
    },
    # === Navigation ===
    "BreadcrumbList": {"itemListElement", "numberOfItems", "itemListOrder"},
    "ListItem": {"item", "nextItem", "position", "previousItem"},
    "ItemList": {"itemListElement", "numberOfItems", "itemListOrder"},
    # === WebSite/WebPage ===
    "WebSite": {"issn", "potentialAction"},
    "WebPage": {
        "breadcrumb", "lastReviewed", "mainContentOfPage", "primaryImageOfPage",
        "relatedLink", "reviewedBy", "significantLink", "speakable",
        "specialty",
    },
    "SearchAction": {"query", "query-input", "target", "result"},
    "ReadAction": {"target"},
    # === Other common types ===
    "HowTo": {
        "estimatedCost", "performTime", "prepTime", "step", "supply",
        "tool", "totalTime", "yield",
    },
    "HowToStep": {"itemListElement", "position", "text"},
    "HowToDirection": {"afterMedia", "beforeMedia", "duringMedia", "performTime", "prepTime", "supply", "text", "tool"},
    "HowToTip": {"text"},
    "NutritionInformation": {
        "calories", "carbohydrateContent", "cholesterolContent", "fatContent",
        "fiberContent", "proteinContent", "saturatedFatContent", "servingSize",
        "sodiumContent", "sugarContent", "transFatContent", "unsaturatedFatContent",
    },
    "ContactPoint": {
        "areaServed", "availableLanguage", "contactOption", "contactType",
        "email", "faxNumber", "hoursAvailable", "productSupported", "telephone",
    },
    "OpeningHoursSpecification": {
        "closes", "dayOfWeek", "opens", "validFrom", "validThrough",
    },
    "Service": {
        "aggregateRating", "areaServed", "audience", "availableChannel",
        "award", "brand", "broker", "category", "hasOfferCatalog",
        "hoursAvailable", "isRelatedTo", "isSimilarTo", "logo", "offers",
        "produces", "provider", "providerMobility", "review",
        "serviceArea", "serviceAudience", "serviceOutput", "serviceType",
        "slogan", "termsOfService",
    },
    "ImageObject": {
        "caption", "contentSize", "contentUrl", "embedUrl", "encodingFormat",
        "exifData", "height", "representativeOfPage", "thumbnail", "width",
    },
    "VideoObject": {
        "actor", "caption", "contentUrl", "director", "duration",
        "embedUrl", "encodingFormat", "height", "musicBy", "productionCompany",
        "thumbnail", "thumbnailUrl", "transcript", "uploadDate",
        "videoFrameSize", "videoQuality", "width",
    },
    "Menu": {"hasMenuSection", "hasMenuItem"},
    "MenuItem": {"menuAddOn", "nutrition", "offers", "suitableForDiet"},
    "MenuSection": {"hasMenuItem", "hasMenuSection"},
    "Course": {"courseCode", "coursePrerequisites", "educationalCredentialAwarded", "hasCourseInstance", "numberOfCredits", "provider"},
    "SoftwareApplication": {
        "applicationCategory", "applicationSubCategory", "applicationSuite",
        "availableOnDevice", "countriesNotSupported", "countriesSupported",
        "downloadUrl", "featureList", "fileSize", "installUrl",
        "memoryRequirements", "operatingSystem", "permissions",
        "processorRequirements", "releaseNotes", "screenshot",
        "softwareAddOn", "softwareHelp", "softwareRequirements",
        "softwareVersion", "storageRequirements",
    },
    "JobPosting": {
        "applicantLocationRequirements", "applicationContact", "baseSalary",
        "datePosted", "directApply", "educationRequirements",
        "eligibilityToWorkRequirement", "employerOverview", "employmentType",
        "employmentUnit", "estimatedSalary", "experienceInPlaceOfEducation",
        "experienceRequirements", "hiringOrganization", "incentiveCompensation",
        "industry", "jobBenefits", "jobImmediateStart", "jobLocation",
        "jobLocationType", "jobStartDate", "occupationalCategory",
        "physicalRequirement", "qualifications", "relevantOccupation",
        "responsibilities", "salaryCurrency", "securityClearanceRequirement",
        "sensoryRequirement", "skills", "specialCommitments", "title",
        "totalJobOpenings", "validThrough", "workHours",
    },
}
# Type hierarchy for property inheritance
TYPE_HIERARCHY: dict[str, list[str]] = {
    "Restaurant": ["FoodEstablishment", "LocalBusiness", "Organization", "Place", "Thing"],
    "Bakery": ["FoodEstablishment", "LocalBusiness", "Organization", "Place", "Thing"],
    "BarOrPub": ["FoodEstablishment", "LocalBusiness", "Organization", "Place", "Thing"],
    "CafeOrCoffeeShop": ["FoodEstablishment", "LocalBusiness", "Organization", "Place", "Thing"],
    "FastFoodRestaurant": ["FoodEstablishment", "LocalBusiness", "Organization", "Place", "Thing"],
    "IceCreamShop": ["FoodEstablishment", "LocalBusiness", "Organization", "Place", "Thing"],
    "FoodEstablishment": ["LocalBusiness", "Organization", "Place", "Thing"],
    "Hotel": ["LodgingBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "Hostel": ["LodgingBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "Motel": ["LodgingBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "BedAndBreakfast": ["LodgingBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "LodgingBusiness": ["LocalBusiness", "Organization", "Place", "Thing"],
    "Dentist": ["MedicalBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "Physician": ["MedicalBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "Pharmacy": ["MedicalBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "Hospital": ["MedicalBusiness", "LocalBusiness", "Organization", "Place", "Thing"],
    "LocalBusiness": ["Organization", "Place", "Thing"],
    "Organization": ["Thing"],
    "Product": ["Thing"],
    "Event": ["Thing"],
    "Article": ["CreativeWork", "Thing"],
    "BlogPosting": ["Article", "CreativeWork", "Thing"],
    "NewsArticle": ["Article", "CreativeWork", "Thing"],
    "Recipe": ["HowTo", "CreativeWork", "Thing"],
    "FAQPage": ["WebPage", "CreativeWork", "Thing"],
    "WebPage": ["CreativeWork", "Thing"],
    "WebSite": ["CreativeWork", "Thing"],
    "Person": ["Thing"],
    "Place": ["Thing"],
    "Offer": ["Thing"],
    "AggregateOffer": ["Offer", "Thing"],
    "Review": ["CreativeWork", "Thing"],
    "AggregateRating": ["Rating", "Thing"],
    "Rating": ["Thing"],
    "Service": ["Thing"],
    "BreadcrumbList": ["ItemList", "Thing"],
    "ItemList": ["Thing"],
    "HowTo": ["CreativeWork", "Thing"],
    "Course": ["CreativeWork", "Thing"],
    "SoftwareApplication": ["CreativeWork", "Thing"],
    "JobPosting": ["Thing"],
    # Add more subtypes following the same pattern from SCHEMA_TYPES keys
}
# Auto-populate missing hierarchy entries for LocalBusiness subtypes
_lb_subtypes = {
    "AutoRepair", "AutoDealer", "GasStation", "HealthClub", "DaySpa",
    "BeautySalon", "HairSalon", "NailSalon", "BarberShop", "Plumber",
    "Electrician", "HVACBusiness", "RoofingContractor", "LockSmith", "Locksmith",
    "MovingCompany", "HousePainter", "GeneralContractor", "Store",
    "ClothingStore", "ElectronicsStore", "FurnitureStore", "GardenStore",
    "GroceryStore", "HardwareStore", "HobbyShop", "HomeGoodsStore",
    "JewelryStore", "LiquorStore", "MensClothingStore", "MobilePhoneStore",
    "MusicStore", "OfficeEquipmentStore", "OutletStore", "PetStore",
    "ShoeStore", "SportingGoodsStore", "TireShop", "ToyStore", "WholesaleStore",
    "AccountingService", "LegalService", "Notary", "Attorney",
    "RealEstateAgent", "TravelAgency", "InsuranceAgency", "EmploymentAgency",
    "FinancialService", "ProfessionalService", "ChildCare",
    "DryCleaningOrLaundry", "Library", "RecyclingCenter", "SelfStorage",
    "ShoppingCenter", "EntertainmentBusiness", "AmusementPark", "Casino",
    "NightClub", "MovieTheater", "BowlingAlley", "ExerciseGym", "GolfCourse",
    "SkiResort", "SportsActivityLocation", "StadiumOrArena", "TennisComplex",
    "GovernmentOffice", "PostOffice", "FireStation", "PoliceStation",
    "AnimalShelter", "Campground", "ComedyClub", "InternetCafe",
    "RadioStation", "TelevisionStation", "TouristInformationCenter",
    "EmergencyService", "MedicalBusiness", "AutomotiveBusiness",
    "HealthAndBeautyBusiness", "HomeAndConstructionBusiness",
}
for _st in _lb_subtypes:
    if _st not in TYPE_HIERARCHY:
        TYPE_HIERARCHY[_st] = ["LocalBusiness", "Organization", "Place", "Thing"]
# Schema.org enumeration values (common ones)
VALID_AVAILABILITY = {
    "https://schema.org/InStock", "https://schema.org/OutOfStock",
    "https://schema.org/PreOrder", "https://schema.org/PreSale",
    "https://schema.org/SoldOut", "https://schema.org/OnlineOnly",
    "https://schema.org/InStoreOnly", "https://schema.org/LimitedAvailability",
    "https://schema.org/Discontinued", "https://schema.org/BackOrder",
}
VALID_ITEM_CONDITION = {
    "https://schema.org/NewCondition", "https://schema.org/UsedCondition",
    "https://schema.org/RefurbishedCondition", "https://schema.org/DamagedCondition",
}
VALID_EVENT_STATUS = {
    "https://schema.org/EventScheduled", "https://schema.org/EventCancelled",
    "https://schema.org/EventMovedOnline", "https://schema.org/EventPostponed",
    "https://schema.org/EventRescheduled",
}
VALID_EVENT_ATTENDANCE_MODE = {
    "https://schema.org/OfflineEventAttendanceMode",
    "https://schema.org/OnlineEventAttendanceMode",
    "https://schema.org/MixedEventAttendanceMode",
}
VALID_DAY_OF_WEEK = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "https://schema.org/Monday", "https://schema.org/Tuesday",
    "https://schema.org/Wednesday", "https://schema.org/Thursday",
    "https://schema.org/Friday", "https://schema.org/Saturday",
    "https://schema.org/Sunday", "PublicHolidays",
}
@dataclass
class ValidationIssue:
    severity: str  # "error", "warning", "info"
    category: str  # "json", "type", "property", "value", "enumeration", "factual"
    message: str
    path: str = ""  # JSON path to the issue, e.g., "$.offers[0].availability"
@dataclass
class ValidationResult:
    valid: bool  # Overall pass/fail
    json_valid: bool = True
    schema_valid: bool = True
    factual_score: float = 1.0  # 0.0 to 1.0
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")
def get_valid_properties(schema_type: str) -> set[str]:
    """Get all valid properties for a type, including inherited properties."""
    props = set()
    # Own properties
    if schema_type in SCHEMA_TYPES:
        props.update(SCHEMA_TYPES[schema_type])
    # Inherited properties
    for parent in TYPE_HIERARCHY.get(schema_type, []):
        if parent in SCHEMA_TYPES:
            props.update(SCHEMA_TYPES[parent])
    return props
def validate_json_structure(raw_output: str) -> tuple[bool, Any, list[ValidationIssue]]:
    """Stage 1: Parse JSON and check basic JSON-LD structure."""
    issues = []
    # Strip whitespace and potential markdown fences
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        issues.append(ValidationIssue("warning", "json", "Output contained markdown code fences"))
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, None, [ValidationIssue("error", "json", f"Invalid JSON: {e}")]
    # Normalise to list
    entities = data if isinstance(data, list) else [data]
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            issues.append(ValidationIssue("error", "json", f"Entity [{i}] is not a JSON object", f"$[{i}]"))
            continue
        if "@type" not in entity:
            issues.append(ValidationIssue("error", "json", f"Entity [{i}] missing @type", f"$[{i}]"))
        if "@context" not in entity:
            issues.append(ValidationIssue("warning", "json", f"Entity [{i}] missing @context", f"$[{i}]"))
        ctx = entity.get("@context", "")
        if ctx and "schema.org" not in str(ctx):
            issues.append(ValidationIssue("error", "json", f"Entity [{i}] @context is not schema.org: {ctx}", f"$[{i}]"))
    has_errors = any(i.severity == "error" for i in issues)
    return not has_errors, data, issues
def validate_schema_vocabulary(data: Any, path: str = "$") -> list[ValidationIssue]:
    """Stage 2: Check types and properties against schema.org vocabulary."""
    issues = []
    entities = data if isinstance(data, list) else [data]
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        current_path = f"{path}[{i}]" if isinstance(data, list) else path
        # Check @type
        schema_type = entity.get("@type")
        if isinstance(schema_type, list):
            # Multi-type entities
            for t in schema_type:
                if t not in SCHEMA_TYPES and t not in TYPE_HIERARCHY:
                    issues.append(ValidationIssue("error", "type", f"Unknown @type: {t}", current_path))
            schema_type = schema_type[0]  # Use first type for property validation
        elif schema_type and schema_type not in SCHEMA_TYPES and schema_type not in TYPE_HIERARCHY:
            issues.append(ValidationIssue("error", "type", f"Unknown @type: {schema_type}", current_path))
        if not schema_type:
            continue
        valid_props = get_valid_properties(schema_type)
        for key, value in entity.items():
            if key.startswith("@"):
                continue
            # Check property exists for this type
            if valid_props and key not in valid_props:
                # Could be valid for a different type — flag as warning not error
                # because our vocabulary isn't exhaustive
                issues.append(ValidationIssue(
                    "warning", "property",
                    f"Property '{key}' may not be valid for {schema_type}",
                    f"{current_path}.{key}"
                ))
            # Validate enumeration values
            if key == "availability" and isinstance(value, str):
                if value not in VALID_AVAILABILITY:
                    issues.append(ValidationIssue(
                        "error", "enumeration",
                        f"Invalid availability value: '{value}'. Must be a schema.org URL like 'https://schema.org/InStock'",
                        f"{current_path}.{key}"
                    ))
            if key == "itemCondition" and isinstance(value, str):
                if value not in VALID_ITEM_CONDITION:
                    issues.append(ValidationIssue(
                        "error", "enumeration",
                        f"Invalid itemCondition: '{value}'",
                        f"{current_path}.{key}"
                    ))
            if key == "eventStatus" and isinstance(value, str):
                if value not in VALID_EVENT_STATUS:
                    issues.append(ValidationIssue(
                        "error", "enumeration",
                        f"Invalid eventStatus: '{value}'",
                        f"{current_path}.{key}"
                    ))
            if key == "eventAttendanceMode" and isinstance(value, str):
                if value not in VALID_EVENT_ATTENDANCE_MODE:
                    issues.append(ValidationIssue(
                        "error", "enumeration",
                        f"Invalid eventAttendanceMode: '{value}'",
                        f"{current_path}.{key}"
                    ))
            if key == "dayOfWeek":
                days = value if isinstance(value, list) else [value]
                for d in days:
                    if isinstance(d, str) and d not in VALID_DAY_OF_WEEK:
                        issues.append(ValidationIssue(
                            "error", "enumeration",
                            f"Invalid dayOfWeek: '{d}'",
                            f"{current_path}.{key}"
                        ))
            # Validate date formats
            if key in ("startDate", "endDate", "datePublished", "dateModified",
                       "dateCreated", "datePosted", "validThrough", "priceValidUntil",
                       "availabilityStarts", "availabilityEnds", "foundingDate", "birthDate"):
                if isinstance(value, str) and value:
                    if not re.match(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?(Z|[+-]\d{2}:\d{2})?)?$", value):
                        issues.append(ValidationIssue(
                            "warning", "value",
                            f"Date '{value}' may not be ISO 8601 format",
                            f"{current_path}.{key}"
                        ))
            # Validate duration formats
            if key in ("prepTime", "cookTime", "totalTime", "performTime", "duration"):
                if isinstance(value, str) and value:
                    if not re.match(r"^P(T(\d+H)?(\d+M)?(\d+S)?|(\d+D)?(T(\d+H)?(\d+M)?(\d+S)?)?)$", value):
                        issues.append(ValidationIssue(
                            "warning", "value",
                            f"Duration '{value}' may not be ISO 8601 format (expected PT...)",
                            f"{current_path}.{key}"
                        ))
            # Validate price has currency
            if key == "price" and isinstance(value, (int, float, str)):
                parent_has_currency = "priceCurrency" in entity
                if not parent_has_currency:
                    issues.append(ValidationIssue(
                        "warning", "value",
                        "price specified without priceCurrency",
                        f"{current_path}.{key}"
                    ))
            # Recurse into nested objects
            if isinstance(value, dict):
                nested_issues = validate_schema_vocabulary(value, f"{current_path}.{key}")
                issues.extend(nested_issues)
            elif isinstance(value, list):
                for j, item in enumerate(value):
                    if isinstance(item, dict):
                        nested_issues = validate_schema_vocabulary(item, f"{current_path}.{key}[{j}]")
                        issues.extend(nested_issues)
    return issues
class TextExtractor(HTMLParser):
    """Extract visible text from HTML for factual cross-referencing."""
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = False
        self._skip_tags = {"script", "style", "noscript", "meta", "link", "head"}
    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._skip = True
    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags:
            self._skip = False
    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.text_parts.append(text)
    def get_text(self) -> str:
        return " ".join(self.text_parts)
def normalise_text(text: str) -> str:
    """Normalise text for fuzzy matching."""
    return re.sub(r"\s+", " ", text.lower().strip())
def extract_values_from_jsonld(data: Any) -> list[tuple[str, str]]:
    """Extract all leaf string values from JSON-LD with their paths."""
    results = []
    def _walk(obj, path="$"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("@"):
                    continue
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")
        elif isinstance(obj, str) and len(obj) > 2:
            # Skip URLs, schema.org references, and very short values
            if not obj.startswith("http") and not obj.startswith("https://schema.org"):
                results.append((path, obj))
    _walk(data)
    return results
def validate_factual_accuracy(data: Any, html_source: str) -> tuple[float, list[ValidationIssue]]:
    """Stage 3: Cross-reference JSON-LD values against HTML source text."""
    issues = []
    # Extract text from HTML
    extractor = TextExtractor()
    try:
        extractor.feed(html_source)
    except Exception:
        return 1.0, [ValidationIssue("warning", "factual", "Could not parse HTML for cross-referencing")]
    html_text = normalise_text(extractor.get_text())
    # Also normalise the raw HTML for attribute matching
    html_raw_normalised = normalise_text(html_source)
    # Extract values from JSON-LD
    values = extract_values_from_jsonld(data)
    if not values:
        return 1.0, []
    found = 0
    checked = 0
    for path, value in values:
        # Skip dates, durations, schema references
        if re.match(r"^\d{4}-\d{2}-\d{2}", value):
            continue
        if re.match(r"^PT?\d", value):
            continue
        checked += 1
        norm_value = normalise_text(value)
        # Check if the value (or a reasonable substring) appears in the HTML
        if len(norm_value) < 3:
            found += 1  # Skip very short values
            continue
        if norm_value in html_text or norm_value in html_raw_normalised:
            found += 1
        else:
            # Try partial matching for longer strings (e.g., descriptions may be truncated)
            words = norm_value.split()
            if len(words) >= 3:
                # Check if at least 60% of words appear in HTML
                word_matches = sum(1 for w in words if len(w) > 2 and w in html_text)
                if word_matches / len(words) >= 0.6:
                    found += 1
                else:
                    issues.append(ValidationIssue(
                        "warning", "factual",
                        f"Value not found in HTML: '{value[:80]}...' " if len(value) > 80 else f"Value not found in HTML: '{value}'",
                        path
                    ))
            else:
                issues.append(ValidationIssue(
                    "warning", "factual",
                    f"Value not found in HTML: '{value}'",
                    path
                ))
    score = found / checked if checked > 0 else 1.0
    return score, issues
def validate(raw_output: str, html_source: str = "") -> ValidationResult:
    """Run the full three-stage validation pipeline.
    Args:
        raw_output: The raw JSON-LD string from the teacher model
        html_source: The original HTML of the page (for factual cross-referencing)
    Returns:
        ValidationResult with pass/fail, scores, and detailed issues
    """
    result = ValidationResult(valid=True)
    # Stage 1: JSON structure
    json_valid, data, json_issues = validate_json_structure(raw_output)
    result.json_valid = json_valid
    result.issues.extend(json_issues)
    if not json_valid or data is None:
        result.valid = False
        result.stats = {"stage_failed": "json_structure"}
        return result
    # Stage 2: Schema vocabulary
    vocab_issues = validate_schema_vocabulary(data)
    result.issues.extend(vocab_issues)
    vocab_errors = sum(1 for i in vocab_issues if i.severity == "error")
    if vocab_errors > 3:
        result.schema_valid = False
    # Stage 3: Factual cross-referencing (only if HTML provided)
    if html_source:
        factual_score, factual_issues = validate_factual_accuracy(data, html_source)
        result.factual_score = factual_score
        result.issues.extend(factual_issues)
    # Compute stats
    entities = data if isinstance(data, list) else [data]
    types_found = []
    for e in entities:
        if isinstance(e, dict):
            t = e.get("@type", "Unknown")
            types_found.append(t if isinstance(t, str) else str(t))
    result.stats = {
        "entity_count": len(entities),
        "types": types_found,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "factual_score": round(result.factual_score, 3),
    }
    # Final pass/fail decision
    result.valid = (
        result.json_valid
        and result.schema_valid
        and result.error_count <= 2  # Allow up to 2 minor errors
        and result.factual_score >= 0.5  # At least 50% of values found in HTML
    )
    return result
# ============================================================================
# CLI interface for batch validation
# ============================================================================
if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Validate schema.org JSON-LD output")
    parser.add_argument("jsonld_file", help="Path to JSON-LD output file")
    parser.add_argument("--html", help="Path to source HTML file for factual cross-referencing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all issues")
    args = parser.parse_args()
    with open(args.jsonld_file) as f:
        raw = f.read()
    html = ""
    if args.html:
        with open(args.html) as f:
            html = f.read()
    result = validate(raw, html)
    status = "PASS" if result.valid else "FAIL"
    print(f"\n{'='*60}")
    print(f"  Validation: {status}")
    print(f"  Entities: {result.stats.get('entity_count', 0)}")
    print(f"  Types: {', '.join(result.stats.get('types', []))}")
    print(f"  Errors: {result.error_count} | Warnings: {result.warning_count}")
    print(f"  Factual score: {result.stats.get('factual_score', 'N/A')}")
    print(f"{'='*60}")
    if args.verbose or not result.valid:
        for issue in result.issues:
            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "?")
            print(f"  {icon} [{issue.category}] {issue.message}")
            if issue.path:
                print(f"     at {issue.path}")
    sys.exit(0 if result.valid else 1)
