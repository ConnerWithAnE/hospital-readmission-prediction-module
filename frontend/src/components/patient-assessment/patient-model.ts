interface PatientData {
    age: number;
    gender: GenderEnum;
    race: RaceEnum;
    time_in_hospital: number;
    admission_type: AdmissionTypeEnum;
    admission_source: AdmissionSourceEnum;
    discharge_group: DischargeGroupEnum;
    num_lab_procedures: number;
    num_procedures: number;
    num_medications: number;
    number_diagnoses: number;
    number_inpatient: number;
    number_outpatient: number;
    number_emergency: number;
}

enum GenderEnum {
    male = "male",
    female = "female"
}

enum RaceEnum {
    african_american = "AfricanAmerican",
    asian = "Asian",
    caucasian = "Caucasian",
    hispanic = "Hispanic",
    other = "Other",
    unknown = "unknown"
}

enum AdmissionSourceEnum {
    physician_referral = "physician_referral",
    emergency = "emergency",
    transfer = "transfer",
    birth = "birth",
    legal = "legal",
    unknown = "unknown",
}

enum DischargeGroupEnum {
    home = "Home",
    transfer = "transfer",
    care_facility = "care_facility",
    hospice_death = "hospice_death",
    other = "Other"
}

enum AdmissionTypeEnum {
    emergency = "emergency",
    urgent = "urgent",
    elective = "elective",
    unknown = "unknown"
}