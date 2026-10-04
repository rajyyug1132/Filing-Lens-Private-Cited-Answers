// Synthetic 9-page course syllabus (original text, no real institution). Used by the e2e test (rendered to a PDF in the test)
// and by bench/general_gate.ts (tunes the GENERAL gate).
export const SYLLABUS_PAGES: string[] = [
  `Operating Systems
Course Code: CS3201
Department of Computer Science and Engineering
Credits: 4
L-T-P: 3-1-0
Total teaching hours: 40 Hours
CIE 50 marks
SEE 50 marks
Prerequisites: Data Structures, Computer Organization
Semester: Fourth`,
  `Course Objectives
1. Explain the structure and services of an operating system.
2. Describe how processes and threads are created, scheduled and synchronized.
3. Analyse deadlock handling strategies and memory management techniques.
4. Compare file system designs and storage allocation methods.
5. Discuss protection and security mechanisms in modern systems.`,
  `Module 1: Introduction and Process Management — 8 Hours
Operating system structure, system calls, kernel and user mode, process concept, process states, process control block, context switch, threads, multithreading models, inter-process communication using shared memory and message passing.`,
  `Module 2: CPU Scheduling and Synchronization — 9 Hours
Scheduling criteria, first come first served, shortest job first, round robin, priority scheduling, multilevel queues, critical section problem, Peterson's solution, semaphores, monitors, classical synchronization problems such as bounded buffer and readers writers.`,
  `Module 3: Deadlocks and Memory Management — 9 Hours
Deadlock characterization, resource allocation graph, prevention, avoidance with the banker's algorithm, detection and recovery, contiguous allocation, paging, segmentation, translation lookaside buffer, page tables and swapping.`,
  `Module 4: Virtual Memory and File Systems — 8 Hours
Demand paging, page replacement algorithms, thrashing, working set model, file concept, access methods, directory structure, allocation methods, free space management, disk scheduling and RAID levels.`,
  `Module 5: Protection, Security and Case Studies — 6 Hours
Goals of protection, access matrix, capability lists, authentication, program threats, cryptography basics, firewalls, and case studies of Linux process scheduling and Windows memory management.`,
  `Course Outcomes
After completing this course the student will be able to:
CO1: Describe operating system services and process abstractions.
CO2: Apply scheduling and synchronization techniques to concurrent programs.
CO3: Evaluate memory management and deadlock handling schemes.
CO4: Compare file system and disk management strategies.
Evaluation Scheme
CIE: two internal tests of 20 marks each and one assignment of 10 marks.
Internal test 1: 14 February 2026
Internal test 2: 28 March 2026
SEE: a three hour written examination of 100 marks, scaled down to 50 marks.`,
  `Textbooks
1. Abraham Silberschatz, Peter Baer Galvin, Greg Gagne, Operating System Concepts, 10th Edition, Wiley.
2. Andrew S. Tanenbaum, Modern Operating Systems, 4th Edition, Pearson.
Reference Books
1. William Stallings, Operating Systems: Internals and Design Principles, 9th Edition, Pearson.
2. Remzi H. Arpaci-Dusseau, Operating Systems: Three Easy Pieces, Arpaci-Dusseau Books.`,
];

export const SYLLABUS_POSITIVES: string[] = [
  'Which textbooks are listed?', 'Which reference books are listed?', 'What are the course objectives?', 'What are the course outcomes?',
  'How many credits is the course?', 'How many hours is Module 3?', 'What does Module 2 cover?', 'How is the course assessed?',
  'How many marks for CIE?', 'How many marks for SEE?', 'What topics are in Module 4?', 'What is the course code?',
  'When is Internal test 1?', "What's the main syllabus?", 'Which topics are covered under deadlocks?', 'What is covered under virtual memory?',
  'Who wrote Operating System Concepts?', 'What are the prerequisites?', 'How many internal tests are there?', 'What is the L-T-P split?',
];

export const SYLLABUS_NEGATIVES: string[] = [
  'What is the DBMS syllabus?', 'Who won the 2018 FIFA World Cup?', "What was Tesla's revenue in 2022?", 'How do I bake sourdough bread?',
  'What is the capital of Australia?', 'Which textbooks are listed for Computer Networks?', 'What are the modules of Machine Learning?',
  'What is the hostel fee?', 'Who is the vice chancellor?', 'What is the stock price of Apple?', 'When is the Diwali holiday?',
  'What is the exam hall seating plan?', 'List the lab experiments for Database Management Systems', 'What is the placement record this year?',
  'How tall is Mount Everest?', 'What is the syllabus for Digital Signal Processing?', 'Who is the head of the mathematics department?',
  'What is the bus timetable?', 'How many goals did Messi score?', 'What is the library membership fee?',
];

export function syllabusHtml(): string {
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  return `<!doctype html><meta charset="utf-8"><style>body{font:14px/1.5 serif}section{page-break-after:always;padding:24px}pre{font:inherit;white-space:pre-wrap}</style>${SYLLABUS_PAGES.map((p) => `<section><pre>${esc(p)}</pre></section>`).join('')}`;
}
