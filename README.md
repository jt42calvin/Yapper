# YouTube Comedy Highlight Generator

*An AI-powered tool that analyzes YouTube videos to extract and create highlight reels of the funniest moments*

## Team Members
- Jacob Tocila
- Nick Roberts

## Project Overview

This project creates a "toy version" of an event/meeting summarizer, specifically focused on identifying and extracting the funniest moments from YouTube videos. Users can input a YouTube link and receive a trimmed highlight reel containing the most humorous parts of the content.

### Core Functionality
- Download YouTube videos and extract audio/transcripts
- Analyze transcripts using AI to identify funny moments
- Detect laughter and ambient audio cues for humor ranking
- Generate trimmed highlight videos of the funniest parts

## Use Case Scenario

**Primary User Story:**
> Two people are in a room, one wants to show a long funny video to their friend. Their friend doesn't have a lot of time and wants to see JUST the funniest part. This tool can be used to help quickly identify the funniest part and generate a short trimmed version of the long video for the busy friend.

## Technical Architecture

### Input Processing
- **Audio Input:** YouTube video URLs
- **Processing Pipeline:**
  - Download video content from YouTube
  - Generate .mp4, .mp3, and .txt files
  - Extract transcripts using speech-to-text APIs:
    - AssemblyAI (cloud-based speech-to-text API)
    - OpenAI Whisper API alternative
  - Analyze audio for laughter detection and intensity ranking

### Output Generation
- **Primary Output:** Trimmed highlight video clips
- **Secondary Output:** Text summaries of funny moments
- **Format Options:** Brief bullet points or full sentences depending on performance requirements

### Current Technology Stack
- Frontend web interface for YouTube URL input
- YouTube data extraction and processing
- Ollama for AI analysis of transcripts
- Audio processing libraries for laughter detection

## Key Design Decisions & Analysis

### Unique Design Choice
**Laughter-Based Humor Detection:** Unlike traditional text summarization systems, our approach incorporates:
- Audio analysis for laughter detection
- Ambient noise processing
- Intensity and duration ranking of laughter
- Multi-modal analysis combining text and audio cues

### Research Questions & Considerations

**Effectiveness & Accuracy:**
- How do we measure what constitutes a "good" funny summary?
- Does the AI actually identify genuinely funny moments?
- How does cultural context affect humor detection?

**Privacy & Ethics:**
- What are the implications of AI-driven content curation?
- Should AI decide what moments are "key" in our conversations?
- How do we handle different types of humor (visual vs. audio)?

**User Experience:**
- Will this tool enhance or replace human judgment in content consumption?
- How does efficiency impact the natural discovery of humor?
- What constitutes an optimal highlight length?

## Current Development Status

### Working Features
✅ YouTube video downloading and processing  
✅ Transcript generation from audio  
✅ Basic AI analysis for humor detection  
✅ File format conversion (.mp4, .mp3, .txt)  
✅ Frontend interface for URL input  

### Known Limitations & Issues
- **Humor Detection Accuracy:** AI often selects setup portions rather than punchlines
  - Example: In John Mulaney content, AI identified opening setup as funniest rather than the actual punchline at the end
- **Cultural Context Limitations:** System may miss niche humor, memes, or culturally specific jokes
- **Bias Concerns:** May be influenced by laugh tracks in low-quality content
- **Limited Scope:** More effective with stand-up comedy format than visual or physical humor

### Systematic Biases & Vulnerabilities
- **Laugh Track Bias:** May incorrectly identify artificially enhanced moments as genuinely funny
- **Cultural Blind Spots:** Training data limitations affect understanding of contemporary humor, memes, and post-ironic content
- **Format Limitations:** Less effective with visual or physical comedy that relies on non-verbal cues
- **Temporal Understanding:** Difficulty distinguishing between setup and payoff in comedic timing

## Evaluation Metrics

### Quantitative Measures
- Accuracy of humor identification vs. human judgment
- User satisfaction with highlight selections
- Time efficiency compared to manual selection
- Consistency across different comedy formats

### User Testing Approach
- Small-scale testing with friends and volunteers
- Feedback collection on highlight quality and usefulness
- Iterative improvement based on user input
- A/B testing of different humor detection algorithms

## Development Roadmap

### Phase 1: Core Functionality (Current)
- ✅ Basic video processing pipeline
- ✅ Transcript generation
- 🔄 Improving humor detection accuracy

### Phase 2: Enhanced Detection
- Audio analysis integration for laughter detection
- Multi-modal analysis combining text and audio
- Cultural context improvement

### Phase 3: User Experience
- Refined frontend interface
- In-browser video player for highlights
- User feedback integration system

### Phase 4: Testing & Iteration
- Comprehensive user testing
- Performance optimization
- Bias mitigation strategies

## Technical Implementation Notes

### API Integration
```
Current APIs in use:
- YouTube data extraction
- Speech-to-text processing
- Ollama for transcript analysis
```

### File Processing Pipeline
```
YouTube URL → Video Download → Audio Extraction → 
Transcript Generation → AI Analysis → Highlight Generation
```

## Future Considerations

### Potential Improvements
- Integration of visual humor detection
- Multi-language support
- Customizable humor preferences
- Integration with existing systems (GoodNotes, FireFlies-style)

### Research Extensions
- Comparative analysis with existing summarization tools
- Investigation of humor subjectivity in AI systems
- Privacy-preserving approaches to content analysis

## Contributing

This project is part of an academic research initiative exploring AI applications in content summarization with a focus on humor detection and human-centered design.

---

*This project serves as both a practical tool and a research platform for understanding how AI can enhance content consumption while maintaining awareness of the social and ethical implications of automated curation systems.*