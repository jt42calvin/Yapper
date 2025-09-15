# yapper

CS 300: AI App, Event Summarizer

---

## Milestone 1: Project proposal, including team members, chosen existing system, and rough outline:

### Project proposal:
Jacob Tocila and Nick Roberts will analyze event summarizing AI applications through interacting with them, and creating a toy version of an event/meeting summarizer. GoodNotes and FireFlies are examples of existing systems that do this. Our "toy example" will record a conversation between multiple people and create an AI summary of the highlights of what is important.

### What questions might you raise in your analysis?

- How will people choose to use this tool?
- How efficient is it? Is it actually helpful?
  - Does it work as intended?
- How does this affect people's privacy? What if they don't want their voice recorded?
  - Do we want to record *everything*?
- Do we want to let AI decide what is a key point in our conversation? 
  - What if AI focuses on something not very useful
- Is it going to replace someone's need for someone to pay attention? 
  - Would this replace someone's ability to take notes and pay attention at work or in class? Our attention span already sucks
- How will this AI app function differently between formal and informal conversations? 
  - Is it going to summarize jokes differently than key points in a meeting?
- What does a good summary look like?
  - How do we measure this?

### What will be the input and output of your toy model?

- Input: audio
  - We'll use some already existing audio processing library
    - AssemblyAI: cloud based speech-to-text cloud API
    - OpenAI Whisper API: an OpenAI version
  - We'll need to take note of how we process 'who said what' since we will use one microphone
- Output: event summary
  - Consider how long we want our summaries to be. Full sentence v. brief incomplete sentences
  - Text bullet points (or maybe full sentences depending on performance)
  - We can evaluate our output through human testing based on this question: "Would this be useful to recall information from a meeting that happened in the past?"

### What situation are you considering for your prototype?

- This will be used when there are meetings between people or in a classroom/tutoring environment.
- A small desktop app that uses the microphone, creates an audio file, uses an already existing cloud API to create a transcript of it, then use an already existing event summarizer API to create bullet point list output
- We can test this by having a short & simple conversation between us/friends who are willing to try this app and give their input (voices & thoughts)
  - This will be part of our user testing & CI/CD
- Our example situations will include conversations between 2 - 5 people, ranging from casual talk to a more purposed meeting.