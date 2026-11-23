"""
sentiment_evaluation_service.py - Student Course Feedback Sentiment Analysis & Model Evaluation.

Features:
- Built-in curated dataset of 240 realistic student course feedback reviews (80 Positive, 80 Neutral, 80 Negative).
- Stratified 80/20 train-test split for reproducible scientific evaluation (192 train, 48 test).
- Scikit-learn TF-IDF feature extraction with ngram_range=(1, 2), sublinear_tf=True, min_df=1.
- 3 Supervised Classifiers:
    1. Multinomial Naive Bayes (MultinomialNB, alpha=0.3)
    2. Logistic Regression (LogisticRegression, C=2.0, max_iter=1000)
    3. Linear Support Vector Machine (LinearSVC, C=1.0)
- Real sklearn.metrics: Accuracy, Weighted Precision, Recall, F1-Score, Per-Class F1, Confusion Matrix.
- Information Extraction: Academic aspect and keyphrase extraction.
- Sarcasm & Error Analysis for academic NLP discussions.
"""

import re
import numpy as np
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

STOP_WORDS = set(ENGLISH_STOP_WORDS)

# ---------------------------------------------------------------------------
# 240 Curated Academic Feedback Samples (80 Positive, 80 Neutral, 80 Negative)
# ---------------------------------------------------------------------------
FEEDBACK_DATASET: List[Dict[str, str]] = [
    # --- POSITIVE (80) ---
    {"text": "The professor explained complex dynamic programming concepts with crystal clear examples and very helpful code walkthroughs.", "label": "Positive"},
    {"text": "Hands-on labs and coding assignments really solidified my understanding of operating systems, excellent practical guidance.", "label": "Positive"},
    {"text": "Excellent course on machine learning. The practical projects helped me land my summer software internship at a great tech firm.", "label": "Positive"},
    {"text": "Great feedback on homework submissions and very approachable teaching assistants during office hours, very helpful support.", "label": "Positive"},
    {"text": "The syllabus is thoroughly updated with modern industry tech stacks like React, FastAPI, and Docker, excellent curriculum design.", "label": "Positive"},
    {"text": "Loved the collaborative group project; building an end-to-end full stack app was an awesome and very rewarding engineering experience.", "label": "Positive"},
    {"text": "Clear grading rubrics and well-structured lecture slides made preparing for exams stress-free, productive, and enjoyable.", "label": "Positive"},
    {"text": "Engaging lectures with active live-coding demonstrations and thoughtful doubt clearing after every module, very helpful professor.", "label": "Positive"},
    {"text": "The course notes and reading materials were comprehensive, clear, and extremely helpful for placement interview preparation.", "label": "Positive"},
    {"text": "Outstanding mentorship from the professor during capstone project milestone reviews and code audits, excellent feedback.", "label": "Positive"},
    {"text": "Thorough coverage of database indexing, normalization, and SQL optimization techniques with real datasets, very helpful practical labs.", "label": "Positive"},
    {"text": "Challenging but deeply rewarding problem sets that significantly boosted my competitive programming skills, great course.", "label": "Positive"},
    {"text": "The flipped classroom model worked wonderfully for computer networks and protocol packet analysis, excellent active learning.", "label": "Positive"},
    {"text": "Super supportive faculty who took extra weekend sessions to explain complex graph theory algorithms, very helpful teaching.", "label": "Positive"},
    {"text": "Practical cloud deployment labs on AWS and Docker containers gave us genuine industry-level confidence, fantastic experience.", "label": "Positive"},
    {"text": "High quality assignments that encouraged clean coding practices, version control, and unit testing, great learning outcomes.", "label": "Positive"},
    {"text": "The guest lectures from industry AI researchers provided fantastic insights into real-world LLM deployments, wonderful exposure.", "label": "Positive"},
    {"text": "Interactive quizzes kept the class engaged and ensured everyone stayed on track with weekly topics, excellent teaching method.", "label": "Positive"},
    {"text": "Best computer science course I have taken so far; the curriculum design is modern, engaging, and intellectually stimulating.", "label": "Positive"},
    {"text": "Detailed code reviews by the TAs helped us write modular, maintainable, and highly efficient algorithms, very helpful advice.", "label": "Positive"},
    {"text": "The professor uses great visual animations to explain binary tree traversals and sorting mechanics, exceptionally clear delivery.", "label": "Positive"},
    {"text": "Very well organized semester schedule with fair deadlines and timely evaluation of lab tasks, outstanding course management.", "label": "Positive"},
    {"text": "Practical security penetration testing exercises made cryptography intuitive rather than purely abstract math, great hands-on experience.", "label": "Positive"},
    {"text": "Excellent credit allocation that generously rewarded student project effort and lab hours, highly appreciated.", "label": "Positive"},
    {"text": "The course forum was actively monitored by instructors, resolving student queries within hours, very responsive and helpful.", "label": "Positive"},
    {"text": "Fantastic balance between theoretical algorithm proofs and real-world Python implementation benchmarks, excellent course.", "label": "Positive"},
    {"text": "Helpful cheat-sheets and review sessions organized before midterm and final semester examinations, very effective preparation.", "label": "Positive"},
    {"text": "Open-ended final project allowed us to explore computer vision models applied to medical healthcare imaging, truly rewarding.", "label": "Positive"},
    {"text": "The professor is passionate about artificial intelligence ethics and sparked meaningful classroom debates, wonderful instructor.", "label": "Positive"},
    {"text": "Clear course roadmap that connected software engineering principles to modern DevOps workflows, excellent practical relevance.", "label": "Positive"},
    {"text": "The instructor took extra time to ensure non-CS students could catch up with object-oriented programming, very supportive and patient.", "label": "Positive"},
    {"text": "Well-equipped computer lab facilities with seamless cloud server access for intensive model training, excellent infrastructure.", "label": "Positive"},
    {"text": "Appreciated the continuous feedback loops and mock coding interviews held before placement season, immensely helpful.", "label": "Positive"},
    {"text": "Every lecture was well paced, starting with quick recaps and ending with practical coding challenges, outstanding structure.", "label": "Positive"},
    {"text": "The compiler design course turned a notoriously daunting subject into an enjoyable, rewarding, and intuitive journey.", "label": "Positive"},
    {"text": "Resourceful textbook recommendations and supplementary research papers enriched our conceptual base, very valuable learning.",
    "label": "Positive"},
    {"text": "Fair credit allocation that accurately reflected the amount of project effort and lab hours required, well appreciated.", "label": "Positive"},
    {"text": "The professor's constructive feedback on my research proposal inspired me to pursue graduate studies, excellent mentor.", "label": "Positive"},
    {"text": "Great integration of Git pull requests and GitHub Classroom for managing group assignment submissions, smooth workflow.", "label": "Positive"},
    {"text": "Insightful case studies of high-scale distributed systems like Netflix microservices and Google Spanner, fantastic lectures.", "label": "Positive"},
    {"text": "Very helpful office hours where the professor patiently clarified all student doubts on concurrency and threads.", "label": "Positive"},
    {"text": "Superb guidance on building scalable microservices with automated CI/CD deployment pipelines, great industry preparation.", "label": "Positive"},
    {"text": "The practical coding assignments were exceptionally well designed and strengthened our algorithmic thinking, excellent labs.", "label": "Positive"},
    {"text": "Remarkable course structure; every theoretical concept was reinforced with live interactive code examples, very clear explanation.", "label": "Positive"},
    {"text": "The instructor provided great real-world analogies that made complex operating system kernels easy and intuitive to understand.", "label": "Positive"},
    {"text": "Highly recommend this course to anyone wanting a deep, rigorous foundation in artificial intelligence and machine learning.", "label": "Positive"},
    {"text": "Teaching assistants were very helpful and provided constructive line-by-line feedback on our code pull requests.", "label": "Positive"},
    {"text": "The professor fostered an encouraging classroom atmosphere where asking foundational questions was welcomed, wonderful teacher.", "label": "Positive"},
    {"text": "Extremely thorough and clear explanations of distributed consensus protocols and Paxos algorithms, great lecture series.", "label": "Positive"},
    {"text": "The semester project allowed our team to deploy a full-scale machine learning model to production seamlessly, rewarding outcome.", "label": "Positive"},
    {"text": "Exceptional quality of lecture recordings and supplementary notes made self-paced revision very effective and smooth.", "label": "Positive"},
    {"text": "The faculty member is a wonderful educator who genuinely cares about every student's academic progress, very inspiring.", "label": "Positive"},
    {"text": "Hands-on database optimization labs helped our team reduce query response times by over eighty percent, great practical lab.", "label": "Positive"},
    {"text": "Clear grading policy and transparent evaluation criteria eliminated unnecessary stress around final exams, excellent fairness.", "label": "Positive"},
    {"text": "Brilliant introduction to natural language processing with modern transformer architectures and embeddings, fascinating course.", "label": "Positive"},
    {"text": "The instructor's passion for algorithmic graph theory was truly contagious and made every lecture exciting and memorable.", "label": "Positive"},
    {"text": "Very responsive discussion board where questions were answered within minutes by instructors and peer mentors, great support.", "label": "Positive"},
    {"text": "Well structured coding challenges that progressively built our confidence from simple arrays to complex trees, great progress.", "label": "Positive"},
    {"text": "The course gave us hands-on experience with modern cloud infrastructure and serverless architecture, highly valuable skills.", "label": "Positive"},
    {"text": "Invaluable career advice and industry interview preparation shared by the professor throughout the semester, very encouraging.", "label": "Positive"},
    {"text": "Extremely engaging teaching style with interactive live coding that kept the entire lecture hall focused and motivated.", "label": "Positive"},
    {"text": "Great balance of theoretical mathematical proofs and hands-on Python data analysis exercises, wonderful course design.", "label": "Positive"},
    {"text": "The capstone project was the highlight of my undergraduate degree, leading directly to a campus placement offer, fantastic.", "label": "Positive"},
    {"text": "Outstanding course materials with clean code templates that made learning web frameworks seamless and enjoyable.", "label": "Positive"},
    {"text": "The professor took proactive steps to ensure students struggling with recursion received extra mentorship, very helpful.", "label": "Positive"},
    {"text": "Comprehensive syllabus covering real-world cybersecurity threats, network firewalls, and cryptographic protocols, great course.", "label": "Positive"},
    {"text": "Very well paced semester schedule with sufficient time allocated between major project milestones, great organization.", "label": "Positive"},
    {"text": "Constructive and detailed grading comments helped us dramatically improve our software engineering practices, excellent feedback.", "label": "Positive"},
    {"text": "The instructor brought immense industry experience into the classroom, illustrating textbook theory with live case studies, superb.", "label": "Positive"},
    {"text": "Inspiring lectures on ethical AI that broadened our perspective on the societal impact of automation, great class discussions.", "label": "Positive"},
    {"text": "Generous lab open hours and high-speed GPU computing access made deep learning experimentation a pleasure, excellent resources.", "label": "Positive"},
    {"text": "The weekly problem sets were intellectually stimulating and prepared us thoroughly for coding competitions, great practice.", "label": "Positive"},
    {"text": "Excellent coordination between lecture topics and weekly laboratory experiments, creating a cohesive curriculum, very satisfying.", "label": "Positive"},
    {"text": "The professor provided fantastic sample exam papers with detailed solution keys that made revision straightforward and clear.", "label": "Positive"},
    {"text": "Fabulous collaborative team experience that mirrored real agile software development sprints in industry, great project.", "label": "Positive"},
    {"text": "Extremely helpful feedback on our mid-term project prototype allowed us to fix critical architectural flaws early, great TA.", "label": "Positive"},
    {"text": "The course notes were so well written they served as my primary reference during technical placement interviews, very helpful.", "label": "Positive"},
    {"text": "Enthusiastic and knowledgeable instructor who made abstract automata theory engaging and accessible, wonderful lectures.", "label": "Positive"},
    {"text": "Very practical curriculum that equipped us with in-demand skills in TypeScript, React, and containerization, highly recommended.", "label": "Positive"},
    {"text": "One of the most rewarding academic courses I have completed, setting a benchmark for university education, exceptional quality.", "label": "Positive"},

    # --- NEUTRAL (80) ---
    {"text": "Course content is standard and covers textbook theory adequately without much deviation, typical academic pace.", "label": "Neutral"},
    {"text": "Pacing of the lectures is average pace; some weeks were packed while other weeks were relatively slow, ordinary course.", "label": "Neutral"},
    {"text": "The assignments are straightforward, average difficulty, neither too challenging nor too elementary for second-year students.", "label": "Neutral"},
    {"text": "Grading followed the standard university rubric without surprises, average exam difficulty and routine marking overall.", "label": "Neutral"},
    {"text": "Slides are provided before class, but practical demonstrations and live coding examples were limited, ordinary lecture style.", "label": "Neutral"},
    {"text": "The textbook is decent, average pace of coverage, though some chapters on legacy microprocessor architecture feel dated.", "label": "Neutral"},
    {"text": "Attendance policy was strictly monitored as per department norms, but lecture delivery was moderate and conventional.", "label": "Neutral"},
    {"text": "Midterm exam was balanced between multiple choice questions and short theoretical derivations, standard difficulty level.", "label": "Neutral"},
    {"text": "The course followed the university syllabus strictly with expected standard topics, typical curriculum progression.", "label": "Neutral"},
    {"text": "Lab sessions covered basic SQL queries; nothing advanced like indexing or query plan tuning was taught, average lab work.", "label": "Neutral"},
    {"text": "Instructor read mostly from the PPT slides, but was willing to answer questions when asked, acceptable delivery.", "label": "Neutral"},
    {"text": "The workload is manageable if you stay consistent, typical for a three-credit engineering subject, average expectations.", "label": "Neutral"},
    {"text": "Prerequisites mentioned in the handbook were accurate, no unexpected math requirements appeared, standard curriculum.", "label": "Neutral"},
    {"text": "Some modules on network protocols were interesting, while physical layer signal theory felt dry, average balance.", "label": "Neutral"},
    {"text": "TA office hours were available twice a week; responses were generally adequate though brief, typical university support.", "label": "Neutral"},
    {"text": "Group project guidelines were standard, allowing teams to pick from a pre-approved list of topics, ordinary scope.", "label": "Neutral"},
    {"text": "Exam questions were directly taken from previous years question papers without modifications, standard exam pattern.", "label": "Neutral"},
    {"text": "Lecture speed was acceptable, average pace, though audio quality in the back benches of the hall could improve.", "label": "Neutral"},
    {"text": "The course provides an acceptable overview of software testing methods without deep automation tools, standard content.", "label": "Neutral"},
    {"text": "Homework deadlines were reasonable and aligned with the central academic calendar, standard semester timeline.", "label": "Neutral"},
    {"text": "A standard introduction to web technologies covering basic HTML, CSS, and elementary JavaScript, moderate depth.", "label": "Neutral"},
    {"text": "The course met general expectations for a core requirement, neither exceptional nor disappointing, ordinary delivery.", "label": "Neutral"},
    {"text": "Lab manuals contained step-by-step procedures which were easy to follow but lacked creative room, standard experiments.", "label": "Neutral"},
    {"text": "Feedback on term papers consisted of basic numerical scores with minimal written comments, typical grading method.", "label": "Neutral"},
    {"text": "Topics on greedy algorithms were covered well, but dynamic programming was concluded quickly, average semester coverage.", "label": "Neutral"},
    {"text": "Classroom discussions occurred occasionally when students initiated questions on current tech trends, moderate engagement.", "label": "Neutral"},
    {"text": "Reference materials were readily accessible in the campus library and departmental shared drive, standard resources.", "label": "Neutral"},
    {"text": "The syllabus covers foundational statistics concepts necessary for basic data analytics, adequate theoretical base.", "label": "Neutral"},
    {"text": "Evaluation was split equally between continuous assessments and end-semester examinations, standard grading structure.", "label": "Neutral"},
    {"text": "Some classes were held online due to scheduling clashes, following standard recording links, typical university logistics.", "label": "Neutral"},
    {"text": "The course structure is similar to standard NPTEL offerings on computer architecture, average depth of content.", "label": "Neutral"},
    {"text": "Programming language chosen was Java 8, which is functional but lacks modern language features, standard laboratory setup.", "label": "Neutral"},
    {"text": "Student presentations were scheduled during the last two weeks with average peer participation, typical seminar format.", "label": "Neutral"},
    {"text": "The instructor followed the prescribed syllabus textbook chapter by chapter in chronological order, conventional pace.", "label": "Neutral"},
    {"text": "Doubt sessions were held prior to midterms, summarizing key formulas and definitions, standard review session.", "label": "Neutral"},
    {"text": "Software tools used in the laboratory were standard open-source packages like Eclipse and MySQL, ordinary setup.", "label": "Neutral"},
    {"text": "Overall an ordinary academic experience that fulfills degree requirements without standout moments, standard course.", "label": "Neutral"},
    {"text": "Content difficulty ramped up moderately towards the second half of the semester as expected, average progression.", "label": "Neutral"},
    {"text": "Project deliverables were checked on schedule with standard departmental marking sheets, routine evaluation.", "label": "Neutral"},
    {"text": "The course serves as an acceptable stepping stone for higher-level elective courses, standard foundational subject.", "label": "Neutral"},
    {"text": "Lecture delivery was average pace, neither too fast nor too slow across the semester modules, standard speed.", "label": "Neutral"},
    {"text": "The syllabus covers standard computer science topics as outlined in the university regulatory handbook, routine content.", "label": "Neutral"},
    {"text": "Course expectations were moderate and aligned with typical undergraduate credit requirements, average workload.", "label": "Neutral"},
    {"text": "Assignments were routine textbook exercises, average difficulty with straightforward expectations, standard homework.", "label": "Neutral"},
    {"text": "The professor followed the slides closely, providing acceptable explanations when prompted by students, ordinary teaching.", "label": "Neutral"},
    {"text": "Grading was fair and followed the predetermined department rubric without noticeable discrepancies, average fairness.", "label": "Neutral"},
    {"text": "The workload was moderate and consistent, requiring about four hours of study per week, standard expectation.", "label": "Neutral"},
    {"text": "Midterm questions were standard numericals and conceptual derivations seen in past year papers, typical test.", "label": "Neutral"},
    {"text": "Lab experiments followed step-by-step documentation, offering basic verification of lecture concepts, average practicals.", "label": "Neutral"},
    {"text": "The course provides an acceptable introductory overview of cloud computing fundamentals, moderate depth of study.", "label": "Neutral"},
    {"text": "Lecture pacing was average pace; theoretical derivations took the expected amount of class time, ordinary speed.", "label": "Neutral"},
    {"text": "Course materials and presentation slides were uploaded to the portal with reasonable regularity, standard logistics.", "label": "Neutral"},
    {"text": "The textbook selected was standard for this subject, covering mandatory departmental guidelines, routine readings.", "label": "Neutral"},
    {"text": "Attendance was marked routinely according to university regulations, with standard lecture delivery, average course.", "label": "Neutral"},
    {"text": "Exam difficulty was moderate, testing memorization and basic algorithmic problem solving, typical examination.", "label": "Neutral"},
    {"text": "The project topic list was standard, allowing groups to implement typical database management systems, ordinary scope.", "label": "Neutral"},
    {"text": "TA responses during tutorial hours were satisfactory, answering basic homework clarification queries, standard help.", "label": "Neutral"},
    {"text": "Content coverage was adequate for an introductory elective, neither deeply advanced nor deficient, average scope.", "label": "Neutral"},
    {"text": "The instructor maintained an average pace throughout the term, completing the syllabus on schedule, standard timing.", "label": "Neutral"},
    {"text": "Practical lab assignments were passable, verifying core programming syntax without advanced challenges, ordinary lab.", "label": "Neutral"},
    {"text": "Evaluation weightage was evenly distributed across quizzes, midterms, and the final examination, standard grading.", "label": "Neutral"},
    {"text": "The course met the standard departmental expectations for a core second-year engineering subject, typical syllabus.", "label": "Neutral"},
    {"text": "Class discussions were ordinary, focusing mostly on clarifying doubts from previous lecture slides, average interaction.", "label": "Neutral"},
    {"text": "Prerequisite knowledge required was average, consistent with basic high school mathematics and C syntax, standard level.", "label": "Neutral"},
    {"text": "Online portal was functional for submitting assignments, with standard automated grading scripts, routine system.", "label": "Neutral"},
    {"text": "The lecture halls were typical, and audio-visual equipment operated with average reliability, ordinary classroom.", "label": "Neutral"},
    {"text": "Syllabus depth was moderate, providing broad exposure rather than specialized domain mastery, standard curriculum.", "label": "Neutral"},
    {"text": "Peer presentations during the final week were satisfactory and adhered to standard time limits, ordinary format.", "label": "Neutral"},
    {"text": "The course structure followed traditional pedagogical methods with regular weekly quizzes, standard academic style.", "label": "Neutral"},
    {"text": "Feedback on homework was limited to letter marks, typical for large enrolled lecture sections, standard turnaround.", "label": "Neutral"},
    {"text": "Course pace remained average pace throughout the semester, covering one textbook chapter every fortnight, typical.", "label": "Neutral"},
    {"text": "The lab manual was structured adequately, providing enough guidance to complete exercises on time, average detail.", "label": "Neutral"},
    {"text": "Examination format was balanced between multiple choice questions and descriptive explanations, standard design.", "label": "Neutral"},
    {"text": "General academic expectations were met, providing a passable foundation for subsequent electives, average course.", "label": "Neutral"},
    {"text": "The instructor answered student questions reasonably well during designated office hours, standard assistance.", "label": "Neutral"},
    {"text": "Workload was fair and did not interfere excessively with other concurrent engineering subjects, average intensity.", "label": "Neutral"},
    {"text": "The curriculum reflected standard university syllabus specifications without contemporary industry additions, routine.", "label": "Neutral"},
    {"text": "Weekly assignments were of medium complexity, taking a predictable amount of effort to submit, standard difficulty.", "label": "Neutral"},
    {"text": "Classroom attendance was recorded via standard bio-metric systems, with normal classroom lectures, ordinary class.", "label": "Neutral"},
    {"text": "Overall a regular academic offering that fulfills curriculum credits with conventional instructional quality, standard.", "label": "Neutral"},

    # --- NEGATIVE (80) ---
    {"text": "The lectures were rushed and not clear, the instructor barely took any questions on graph traversal algorithms, poor delivery.", "label": "Negative"},
    {"text": "Outdated curriculum focusing on deprecated tools with completely inadequate lab practice, very disappointing course.", "label": "Negative"},
    {"text": "Exams were disproportionately hard and not clear compared to what was covered in class slides, unfair evaluation.", "label": "Negative"},
    {"text": "Assignment guidelines were vague, not clear, and grading took over a month with no constructive feedback, terrible experience.", "label": "Negative"},
    {"text": "The online lab environment crashed constantly during timed practical exams, causing huge distress and unfair mark deductions.", "label": "Negative"},
    {"text": "Monotonous lectures where the instructor simply read verbatim from twenty-year-old PowerPoint slides, unengaging and boring.", "label": "Negative"},
    {"text": "Unreasonable workload with overlapping deadlines that completely disregarded other major core courses, chaotic and stressful.", "label": "Negative"},
    {"text": "TAs were unresponsive to email queries and frequently gave contradictory grading standards, very unhelpful teaching assistants.", "label": "Negative"},
    {"text": "Heavy emphasis on rote memorization of definitions rather than understanding underlying system design, poor pedagogy.", "label": "Negative"},
    {"text": "Disorganized course syllabus with sudden changes to evaluation weightages two weeks before finals, totally unacceptable.", "label": "Negative"},
    {"text": "Extremely harsh and opaque grading on the term project with no breakdown of deducted points, completely unfair marking.", "label": "Negative"},
    {"text": "Lab machines were poorly maintained, missing necessary C++ compilers and Python libraries, completely neglected facility.", "label": "Negative"},
    {"text": "The professor frequently canceled scheduled lectures and then dumped marathon makeup classes on weekends, terrible scheduling.", "label": "Negative"},
    {"text": "No practical relevance to modern industry standards; taught obsolete technologies no company uses, useless curriculum.", "label": "Negative"},
    {"text": "Course prerequisite requirements were totally misleading and not clear, leaving students stranded without foundational math.", "label": "Negative"},
    {"text": "Feedback on code submissions was non-existent; we only received a bare letter grade weeks later, zero guidance provided.", "label": "Negative"},
    {"text": "Classroom environment was unengaging and student questions were routinely dismissed as silly or off-topic, hostile professor.", "label": "Negative"},
    {"text": "Project requirements were altered midway through the semester, invalidating three weeks of our team's work, disorganized.", "label": "Negative"},
    {"text": "Theoretical derivations took up ninety percent of class time while coding implementation was ignored, completely unbalanced.", "label": "Negative"},
    {"text": "Unfair exam distribution where eighty percent of marks tested minor footnotes from unassigned readings, terrible test.", "label": "Negative"},
    {"text": "The syllabus is bloated with redundant theory modules that add no tangible engineering skill, poor course design.", "label": "Negative"},
    {"text": "Instructor arrived twenty minutes late consistently and rushed through complex mathematical proofs, very unprofessional.", "label": "Negative"},
    {"text": "Lab assistant lacked basic knowledge of Linux permissions and was unable to assist students during practicals, unhelpful.", "label": "Negative"},
    {"text": "No mock tests or sample questions were provided prior to a very tricky and unfair end-semester examination, poor preparation.", "label": "Negative"},
    {"text": "Cheating and plagiarism during online programming quizzes went completely unchecked by course proctors, unfair environment.", "label": "Negative"},
    {"text": "Textbook assigned was out of print and university bookstore had zero copies available for checkout, unorganized logistics.", "label": "Negative"},
    {"text": "Communication from course coordinators was chaotic with announcements posted midnight before deadlines, stressful experience.", "label": "Negative"},
    {"text": "Terrible pacing; two months spent on introductory basics and five days crammed with advanced topics, awful planning.", "label": "Negative"},
    {"text": "Hardware kits provided for the embedded systems laboratory were defective and burned out during experiments, broken equipment.", "label": "Negative"},
    {"text": "The course felt like an unnecessary obstacle rather than an educational learning experience, frustrating and poorly taught.", "label": "Negative"},
    {"text": "Subjective grading on essay-style questions with no rubric resulted in inconsistent marks across sections, totally unfair.", "label": "Negative"},
    {"text": "Instructor showed zero empathy towards students facing medical emergencies during midterm exams, very callous behavior.", "label": "Negative"},
    {"text": "Coding starter repository had broken test suites and syntax errors that students had to fix themselves, waste of time.", "label": "Negative"},
    {"text": "Excessive penalty deductions for minor formatting discrepancies in laboratory documentation reports, harsh and pedantic.", "label": "Negative"},
    {"text": "Lectures lacked depth and merely scratched surface level bullet points easily found on Wikipedia, superficial instruction.", "label": "Negative"},
    {"text": "Zero industry exposure or guest seminars, completely disconnected from contemporary engineering practice, obsolete course.", "label": "Negative"},
    {"text": "The final exam was impossibly long and over half the class could not finish writing their solutions, poorly calibrated.", "label": "Negative"},
    {"text": "Instructor refused to re-evaluate obvious grading errors in the midterm paper tabulation, completely stubborn and unfair.", "label": "Negative"},
    {"text": "Poorly planned curriculum where practical lab sessions were completely out of sync with theory lectures, disjointed structure.", "label": "Negative"},
    {"text": "Frustrating experience with unresponsive administration and unhelpful course coordinators, worst course of the degree.", "label": "Negative"},
    {"text": "Lecture explanations were not clear, leaving most students confused about fundamental pointer mechanics, poor teaching.", "label": "Negative"},
    {"text": "The professor was consistently late and rushed through important distributed systems topics, very careless instruction.", "label": "Negative"},
    {"text": "Assignment instructions were not clear and grading rubrics were changed after submission deadlines, highly unfair.", "label": "Negative"},
    {"text": "Extremely poor communication from teaching assistants who ignored student queries on the discussion forum, totally unhelpful.", "label": "Negative"},
    {"text": "The course syllabus is severely outdated and teaches obsolete programming practices from decades ago, poor elective.", "label": "Negative"},
    {"text": "Exams were unreasonably difficult with questions focusing on obscure details never taught in class, unfair paper.", "label": "Negative"},
    {"text": "Terrible classroom experience with monotonous lectures where the instructor merely read slides, unengaging and dull.", "label": "Negative"},
    {"text": "Lab hardware was constantly malfunctioning and instructors were unable to troubleshoot basic errors, defective equipment.", "label": "Negative"},
    {"text": "Grading was completely unfair and lacked transparency, with huge mark deductions for trivial issues, harsh evaluator.", "label": "Negative"},
    {"text": "Workload was excessive and chaotic, with three major project deliverables scheduled on the same date, terrible coordination.", "label": "Negative"},
    {"text": "The instructor dismissed student questions rudely and showed zero patience during office hours, hostile attitude.", "label": "Negative"},
    {"text": "Course materials were disorganized with broken hyperlinks and missing chapter references throughout, careless maintenance.", "label": "Negative"},
    {"text": "Theoretical lectures had zero connection to practical labs, leaving students unprepared for projects, disjointed syllabus.", "label": "Negative"},
    {"text": "Feedback on assignments was not clear and grades were uploaded weeks after finals were already over, irresponsible grading.", "label": "Negative"},
    {"text": "The textbook assigned was completely unhelpful and contained numerous mathematical errors in formulas, bad reference.", "label": "Negative"},
    {"text": "Disastrous course organization with sudden syllabus modifications announced days before the mid-term, chaotic mess.", "label": "Negative"},
    {"text": "Online examination software crashed repeatedly, wiping out submitted answers without recovery options, catastrophic exam.", "label": "Negative"},
    {"text": "The instructor spent eighty percent of lecture time reading definitions rather than solving problems, ineffective teaching.", "label": "Negative"},
    {"text": "Coding starter files had syntax errors and unconfigured dependencies that wasted hours of student time, careless preparation.", "label": "Negative"},
    {"text": "Very disappointing elective that failed completely to teach usable engineering frameworks, frustrating waste of time.", "label": "Negative"},
    {"text": "The teaching was so rushed that complex dynamic programming algorithms were skimmed in thirty minutes, poor explanation.", "label": "Negative"},
    {"text": "Arbitrary and subjective grading on term papers with zero rubric breakdown or justification provided, completely unfair.", "label": "Negative"},
    {"text": "Lab assistants were completely unhelpful and could not answer basic questions on Linux permissions, untrained staff.", "label": "Negative"},
    {"text": "The course was a frustrating waste of time with no practical learning or usable engineering skills gained, very bad experience.", "label": "Negative"},
    {"text": "Unrealistic project scope given the complete absence of mentorship or technical guidance from faculty, zero support.", "label": "Negative"},
    {"text": "The professor canceled multiple lectures without notice and then dumped makeup classes on Sunday mornings, terrible schedule.", "label": "Negative"},
    {"text": "Examination questions tested rote memorization of verbatim slide text instead of conceptual understanding, flawed testing.", "label": "Negative"},
    {"text": "Communication was terrible, with critical project clarifications posted at midnight before deadlines, stressful and chaotic.", "label": "Negative"},
    {"text": "Course prerequisite details were misleading and not clear, leaving many students completely overwhelmed, poor guidance.", "label": "Negative"},
    {"text": "Zero doubt clearing sessions were conducted despite repeated requests from struggling students, uncaring faculty.", "label": "Negative"},
    {"text": "The pacing was completely uneven; two months on elementary syntax and three days on complex trees, terribly rushed.", "label": "Negative"},
    {"text": "Plagiarism in homework went completely unchecked while honest students were penalized for formatting, unfair evaluation.", "label": "Negative"},
    {"text": "The instructor showed utter indifference to student feedback and refused to review grading mistakes, completely unreasonable.", "label": "Negative"},
    {"text": "Hardware experimental kits were broken and burned out components were never replaced by the department, defective labs.", "label": "Negative"},
    {"text": "Lecture slides were full of errors and typos that created enormous confusion during exam preparation, sloppy slides.", "label": "Negative"},
    {"text": "Extremely harsh penalty deductions for minor file naming mistakes completely demoralized the class, punitive environment.", "label": "Negative"},
    {"text": "The course felt punitive rather than educational, with unfair grading policies and hostile delivery, dreadful experience.", "label": "Negative"},
    {"text": "TA office hours were frequently canceled without warning, leaving students stranded before submissions, unhelpful TAs.", "label": "Negative"},
    {"text": "No sample solutions or past exam papers were provided for an extraordinarily tricky final paper, zero preparation help.", "label": "Negative"},
    {"text": "Overall a dreadful academic experience plagued by poor administration, hostile attitude, and zero guidance, terrible course.", "label": "Negative"},
]

# Academic aspect mapping for Information Extraction
ACADEMIC_ASPECTS = {
    "Teaching & Pedagogy": [
        "professor", "instructor", "teaching", "lectures", "explanation", "pedagogy",
        "faculty", "doubt", "slides", "classroom", "delivery", "pacing", "ta", "mentor"
    ],
    "Labs & Hands-on Practice": [
        "lab", "practical", "coding", "hands-on", "exercise", "implementation",
        "linux", "python", "java", "c++", "compiler", "machines", "environment"
    ],
    "Curriculum & Syllabus": [
        "curriculum", "syllabus", "content", "topics", "industry", "modern",
        "outdated", "theory", "textbook", "prerequisite", "algorithms", "frameworks"
    ],
    "Assignments & Projects": [
        "assignment", "project", "homework", "submission", "github", "milestone",
        "capstone", "problem set", "deadlines", "workload", "deliverables"
    ],
    "Exams, Grading & Assessment": [
        "exam", "grading", "marks", "rubric", "midterm", "finals", "scoring",
        "quiz", "evaluation", "questions", "difficulty", "fairness"
    ]
}


class SentimentEvaluationService:
    """
    Manages model training, evaluation, cross-model inference,
    aspect information extraction, and sarcasm analysis.
    """

    def __init__(self):
        self.classes = ["Positive", "Neutral", "Negative"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.models: Dict[str, Any] = {
            "Multinomial Naive Bayes": MultinomialNB(alpha=0.3),
            "Logistic Regression": LogisticRegression(C=2.0, max_iter=1000, random_state=42),
            "Linear Support Vector Machine": LinearSVC(C=1.0, random_state=42)
        }
        self.metrics: Dict[str, Any] = {}
        self.best_model_name: str = ""
        self.is_trained: bool = False
        self._train_and_evaluate()

    def _train_and_evaluate(self):
        """Train on stratified 80/20 split and record comprehensive sklearn metrics."""
        texts = [d["text"] for d in FEEDBACK_DATASET]
        labels = [d["label"] for d in FEEDBACK_DATASET]

        # Stratified 80/20 split (192 train, 48 test)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels
        )

        # Fit TF-IDF on training set
        X_train = self.vectorizer.fit_transform(X_train_raw)
        X_test = self.vectorizer.transform(X_test_raw)

        model_results = {}
        best_score = -1.0
        best_name = ""
        best_cm = []

        for name, clf in self.models.items():
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = float(accuracy_score(y_test, preds))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, preds, average="weighted", zero_division=0
            )

            # Per-class metrics
            per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
                y_test, preds, labels=self.classes, average=None, zero_division=0
            )

            cm = confusion_matrix(y_test, preds, labels=self.classes).tolist()

            model_results[name] = {
                "name": name,
                "accuracy": round(acc, 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "per_class": {
                    cls: {
                        "precision": round(float(per_class_p[i]), 4),
                        "recall": round(float(per_class_r[i]), 4),
                        "f1_score": round(float(per_class_f1[i]), 4),
                        "support": int(per_class_sup[i])
                    }
                    for i, cls in enumerate(self.classes)
                },
                "confusion_matrix": cm
            }

            if f1 > best_score:
                best_score = f1
                best_name = name
                best_cm = cm

        self.metrics = {
            "dataset_size": len(FEEDBACK_DATASET),
            "train_size": len(X_train_raw),
            "test_size": len(X_test_raw),
            "class_distribution": {
                "Positive": labels.count("Positive"),
                "Neutral": labels.count("Neutral"),
                "Negative": labels.count("Negative"),
            },
            "models": model_results,
            "best_model": best_name,
            "best_confusion_matrix": {
                "classes": self.classes,
                "matrix": best_cm
            }
        }
        self.best_model_name = best_name
        self.is_trained = True

    def get_evaluation_overview(self) -> Dict[str, Any]:
        """Return the precomputed evaluation metrics and comparison matrix."""
        if not self.is_trained:
            self._train_and_evaluate()
        return self.metrics

    def analyze_feedback(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment across all 3 models, compute confidence,
        perform Information Extraction (aspects/entities), and analyze sarcasm/errors.
        """
        if not text or not text.strip():
            return {
                "error": "Feedback text cannot be empty",
                "text": text,
                "predictions": {},
                "extracted_aspects": [],
                "sarcasm_analysis": None
            }

        if not self.is_trained:
            self._train_and_evaluate()

        vec = self.vectorizer.transform([text])
        predictions = {}

        for name, clf in self.models.items():
            pred = clf.predict(vec)[0]
            # Probabilities or decision distance
            conf = 0.85
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(vec)[0]
                idx = list(clf.classes_).index(pred)
                conf = float(probs[idx])
            elif hasattr(clf, "decision_function"):
                decision = clf.decision_function(vec)[0]
                if hasattr(decision, "__len__"):
                    idx = list(clf.classes_).index(pred)
                    exp_dec = np.exp(decision - np.max(decision))
                    softmax = exp_dec / np.sum(exp_dec)
                    conf = float(softmax[idx])
                else:
                    conf = float(1 / (1 + np.exp(-abs(decision))))

            predictions[name] = {
                "sentiment": pred,
                "confidence": round(conf, 4)
            }

        # Consensus prediction (majority vote)
        all_preds = [p["sentiment"] for p in predictions.values()]
        from collections import Counter
        consensus = Counter(all_preds).most_common(1)[0][0]

        # Information Extraction: Aspects & Keyphrases
        extracted_aspects = self._extract_aspects_and_phrases(text)

        # Sarcasm / Error Analysis
        sarcasm_meta = self._detect_sarcasm_and_nuance(text, consensus)

        return {
            "text": text,
            "consensus_sentiment": consensus,
            "model_predictions": predictions,
            "best_model_prediction": predictions.get(self.best_model_name, {}).get("sentiment", consensus),
            "extracted_aspects": extracted_aspects,
            "sarcasm_analysis": sarcasm_meta
        }

    def _extract_aspects_and_phrases(self, text: str) -> List[Dict[str, Any]]:
        """Information Extraction: Extract key academic aspects, topics, and relevant snippet phrases."""
        t_lower = text.lower()
        extracted = []

        for aspect_name, keywords in ACADEMIC_ASPECTS.items():
            matched_keywords = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', t_lower)]
            if matched_keywords:
                # Find matching phrase/sentence
                snippet = ""
                sentences = re.split(r'[.!?]+', text)
                for s in sentences:
                    s_clean = s.strip()
                    if any(kw in s_clean.lower() for kw in matched_keywords):
                        snippet = s_clean
                        break

                extracted.append({
                    "aspect": aspect_name,
                    "matched_keywords": matched_keywords[:4],
                    "evidence_snippet": snippet or text[:90] + "..."
                })

        # Key academic n-grams / phrases extraction
        cleaned_words = [w for w in re.sub(r'[^a-zA-Z\s]', ' ', text).split() if len(w) > 3 and w.lower() not in STOP_WORDS]
        bigrams = []
        for i in range(len(cleaned_words) - 1):
            bigrams.append(f"{cleaned_words[i]} {cleaned_words[i+1]}")

        return {
            "aspects": extracted,
            "salient_phrases": bigrams[:5] if bigrams else cleaned_words[:5]
        }

    def _detect_sarcasm_and_nuance(self, text: str, predicted_sentiment: str) -> Dict[str, Any]:
        """
        Heuristic error & linguistic nuance detector:
        Flags sarcasm, contrastive conjunctions, and deceptive positive words in negative context.
        """
        t_lower = text.lower()
        sarcasm_cues = [
            "loved spending", "great job giving", "brilliant idea to", "thanks for",
            "wonderful experience of failing", "so glad we had to", "fascinating to wait",
            "enjoyed staying up all night", "nothing better than crashing"
        ]
        contrast_cues = ["but", "however", "although", "though", "yet", "on the other hand"]
        negation_cues = ["not bad", "not terrible", "never failed", "hardly disappointed"]

        is_sarcastic_candidate = any(cue in t_lower for cue in sarcasm_cues)
        has_contrast = any(re.search(r'\b' + re.escape(c) + r'\b', t_lower) for c in contrast_cues)
        has_negation = any(neg in t_lower for neg in negation_cues)

        explanation = ""
        category = "Standard Direct Expression"

        if is_sarcastic_candidate:
            category = "Potential Sarcasm / Irony Detected"
            explanation = (
                "Surface level vocabulary contains positive polarity tokens (e.g., 'loved', 'great', 'brilliant') "
                "used ironically in a stressful or frustrating context. Traditional TF-IDF bag-of-words models "
                "can misclassify this as Positive unless contextual embeddings or sarcastic nuance patterns are modeled."
            )
        elif has_contrast:
            category = "Mixed / Contrastive Sentiment"
            explanation = (
                "Sentence contains contrastive conjunctions ('but', 'however', 'although') that shift polarity "
                "mid-sentence (e.g. good professor but unfair exams). Model balances opposing token weights."
            )
        elif has_negation:
            category = "Double Negation / Attenuated Sentiment"
            explanation = (
                "Phrase utilizes negation modifiers ('not bad', 'hardly disappointed') which reverse negative words "
                "into mild positive or neutral expressions."
            )
        else:
            explanation = "Direct semantic sentiment alignment. No contradictory irony or syntax inversion observed."

        return {
            "category": category,
            "flagged": is_sarcastic_candidate or has_contrast or has_negation,
            "explanation": explanation,
            "linguistic_markers": {
                "sarcasm_candidate": is_sarcastic_candidate,
                "contrastive_conjunctions": has_contrast,
                "negation_reversal": has_negation
            }
        }


# Singleton instance for high-speed cached inference
sentiment_evaluator = SentimentEvaluationService()
